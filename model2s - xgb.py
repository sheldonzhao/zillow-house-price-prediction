'''
update: log and statistic of top features
res: 0.06450
'''
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import utils
import zipfile
import math
from sklearn.model_selection import GridSearchCV

properties = pd.read_csv(utils.file_properties)
train = pd.read_csv(utils.file_train)

# statistic features type
table_type = properties.dtypes.reset_index()
table_type.columns = ["feature", "Type"]
print(table_type.groupby("Type").aggregate('count'))

print('start to preprocess')
for c in properties.columns:
    if properties[c].dtype == 'float64':
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

train_df = train.merge(properties, how='left', on='parcelid')
x_train = train_df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
# create test set
x_test = properties.drop(['parcelid'], axis=1)
# shape
print('Shape train: {}'.format(x_train.shape))

# drop out ouliers
UP_LIMIT_BODER = 97
DOWN_LIMIT_BODER = 3
ulimit = np.percentile(train.logerror.values, UP_LIMIT_BODER)
llimit = np.percentile(train.logerror.values, DOWN_LIMIT_BODER)
print('the logerror = %f < %f percent' % (ulimit, UP_LIMIT_BODER))
print('the logerror = %f > %f percent' % (llimit, DOWN_LIMIT_BODER))
train_df = train_df[train_df.logerror >= llimit]
train_df = train_df[train_df.logerror <= ulimit]
print('preprocess end')

print('start feature engineering')
statistic_feats = ['taxamount', 'landtaxvaluedollarcnt', 'structuretaxvaluedollarcnt',
                   'lotsizesquarefeet', 'calculatedfinishedsquarefeet',
                   'taxvaluedollarcnt']
median_log_error = train_df['logerror'].median()
# add statistic feature
for c in statistic_feats:
    median = train_df.loc[train_df[c].notnull()][c].median()
    num = int(max(train_df[c]))
    n = len(str(num))
    train_df['temp'] = train_df[c].map(lambda x: np.log(x) // 5)
    group_feat = train_df.groupby(['temp'])['logerror'].mean().to_dict()
    train_df[c + '_logerror'] = train_df['temp'].map(lambda x: group_feat[x])
    x_test['temp'] = x_test[c].map(lambda x: np.log(x) // 5)
    x_test[c + '_logerror'] = x_test['temp'].map(
        lambda x: group_feat[x] if x in group_feat.keys() else median_log_error)

train_df.pop('temp')
x_test.pop('temp')

# add feature about transaction date
train_df["month"] = train_df.transactiondate.map(lambda x: str(x).split("-")[1])
traingroupedMonth = train_df.groupby(["month"])["logerror"].mean().to_frame().reset_index()
train_df['month_logerror'] = train_df['month'].map(lambda x: round(traingroupedMonth.ix[int(x) - 1]['logerror'], 5))
train_df.pop('month')

# create training set
x_train = train_df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print('feature engineering end')
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

# xgboost params
print('start to build model')

xgb_params = {
    'learning_rate': 0.03,
    'max_depth': 5,
    'min_child_weight': 8,
    'silent': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 1000,
    'gamma': 0,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'base_score': y_mean,
}

dtrain = xgb.DMatrix(x_train, y_train)

# cross-validation
cv_result = xgb.cv(xgb_params,
                   dtrain,
                   nfold=5,
                   num_boost_round=1000,
                   early_stopping_rounds=50,
                   verbose_eval=1,
                   show_stdv=False
                   )
num_boost_rounds = len(cv_result)
print(num_boost_rounds)
# train model
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
res = []
for i in range(3):
    x_test['month_logerror'] = round(traingroupedMonth.ix[9 + int(i)]['logerror'], 5)
    dtest = xgb.DMatrix(x_test)
    pred = model.predict(dtest)

    y_pred = []
    for i, predict in enumerate(pred):
        y_pred.append(str(round(predict, 4)))
    res.append(y_pred)

output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
                       '201610': np.array(res[0]), '201611': np.array(res[1]), '201612': np.array(res[2]),
                       '201710': np.array(res[0]), '201711': np.array(res[1]), '201712': np.array(res[2])})
# set col 'ParceID' to first col
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]
output.to_csv(utils.file_output, index=False)
# zip
f = zipfile.ZipFile('output.zip', 'w', zipfile.ZIP_DEFLATED)
f.write('output.csv')
