import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import utils
import zipfile
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

properties = pd.read_csv(utils.file_properties)
train = pd.read_csv(utils.file_train)

print('start to preprocess')
for c in properties.columns:
    if properties[c].dtype == np.float64:
        properties[c].fillna(properties[c].median(), inplace=True)
        properties[c] = properties[c].astype(np.float32)
    if properties[c].dtype == 'object':
        properties[c].fillna(0, inplace=True)
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

feat = ['propertycountylandusecode', 'fireplaceflag', 'taxdelinquencyflag']
for c in feat:
    feat_df = pd.get_dummies(properties[c], prefix=c)
    properties = pd.concat([properties, feat_df], axis=1)
    properties.pop(c)

# cross features
print('preprocess ends ')
train_df = train.merge(properties, how='left', on='parcelid')
print(train_df.shape)

# add statistic feature
print('start feature engineering')
train_df['transactiondate'] = pd.to_datetime(train_df['transactiondate'])
train_df["month"] = train_df.transactiondate.dt.month
train_df["quarter"] = train_df.transactiondate.dt.quarter
traingroupedMonth = train_df.groupby(["month"])["logerror"].mean().to_frame().reset_index()
traingroupedQuarter = train_df.groupby(["quarter"])["logerror"].mean().to_frame().reset_index()
train_df['month_logerror'] = train_df['month'].map(lambda x: round(traingroupedMonth.ix[int(x) - 1]['logerror'], 6))
train_df['quarter_logerror'] = train_df['quarter'].map(
    lambda x: round(traingroupedQuarter.ix[int(x) - 1]['logerror'], 6))
train_df.pop('quarter')

select_quarter4 = train_df['month'] > 9
train_df.pop('month')

# create training set
x_train_all = train_df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
y_mean = np.mean(train_df["logerror"])
# create test set
x_test = properties.drop(['parcelid'], axis=1)

print('After removing outliers:')
print('Shape train: {}'.format(x_train_all.shape))
print('feature engineering end')

# xgboost params
print('start to build model')
xgb_params = {
    'learning_rate': 0.07,
    'max_depth': 6,
    'silent': 1,
    'alpha': 0.8,
    'lambda': 8,
    'subsample': 0.5,
    'colsample_bytree': 0.7,
    'n_estimators': 1000,
    'gamma': 0,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'base_score': y_mean,
}

x_valid = x_train_all[select_quarter4]
y_valid = train_df['logerror'].values.astype(np.float32)[select_quarter4]
dvalid_x = xgb.DMatrix(x_valid)
dvalid_xy = xgb.DMatrix(x_valid, y_valid)

# drop out ouliers
train_df = train_df[~select_quarter4][(train_df.logerror >= -0.4) & (train_df.logerror <= 0.419)]
y_train = train_df["logerror"].values.astype(np.float32)
x_train = train_df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)

dtrain = xgb.DMatrix(x_train, y_train)
# train model
evals = [(dtrain, 'train'), (dvalid_xy, 'eval')]
model = xgb.train(xgb_params, dtrain, num_boost_round=2000, evals=evals, early_stopping_rounds=100, verbose_eval=10)
# valid set
valid_pred = model.predict(dvalid_x)
print(mean_absolute_error(y_valid, valid_pred))

res = []
for i in range(3):
    x_test['month_logerror'] = round(traingroupedMonth.ix[9 + int(i)]['logerror'], 6)
    x_test['quarter_logerror'] = round(traingroupedQuarter.ix[3]['logerror'], 6)
    dtest = xgb.DMatrix(x_test)
    pred = model.predict(dtest)
    res.append(pred)

output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
                       '201610': res[0], '201611': res[1], '201612': res[2],
                       '201710': res[0], '201711': res[1], '201712': res[2]})
# set col 'ParceID' to first col
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]
output.to_csv(utils.file_output, index=False, float_format='%.6f')
# zip
f = zipfile.ZipFile('output.zip', 'w', zipfile.ZIP_DEFLATED)
f.write('output.csv')
