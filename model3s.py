import numpy as np
import pandas as pd
import utils
import zipfile
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

properties = pd.read_csv(utils.file_properties)
train = pd.read_csv(utils.file_train)

print('start to preprocess')
for c in properties.columns:
    properties[c] = properties[c].fillna(0)
    if properties[c].dtype == 'object':
        properties.pop(c)

train_df = train.merge(properties, how='left', on='parcelid')
x_test = properties.drop(['parcelid'], axis=1)

# add statistic feature
train_df["month"] = train_df.transactiondate.map(lambda x: str(x).split("-")[1])
traingroupedMonth = train_df.groupby(["month"])["logerror"].mean().to_frame().reset_index()
train_df['month_logerror'] = train_df['month'].map(lambda x: round(traingroupedMonth.ix[int(x) - 1]['logerror'], 5))
train_df.pop('month')

# drop out ouliers
UP_LIMIT_BODER = 97
DOWN_LIMIT_BODER = 3
ulimit = np.percentile(train.logerror.values, UP_LIMIT_BODER)
llimit = np.percentile(train.logerror.values, DOWN_LIMIT_BODER)
print('the logerror = %f < %f percent' % (ulimit, UP_LIMIT_BODER))
print('the logerror = %f > %f percent' % (llimit, DOWN_LIMIT_BODER))
train_df = train_df[train_df.logerror >= llimit]
train_df = train_df[train_df.logerror <= ulimit]

x_train = train_df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print('After removing outliers:')
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

# model
'''
params = [4, 5, 6, 7]
test_scores = []
for param in params:
    xgb = XGBRegressor(max_depth=7, learning_rate=0.05, base_score=y_mean, min_child_weight=7, subsample=0.8,
                       colsample_bytree=0.8, n_estimators=200)
    test_score = np.sqrt(-cross_val_score(xgb, x_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
plt.plot(params, test_scores)
plt.title('xgboost error')
plt.show()
'''
res = []
for i in range(3):
    x_test['month_logerror'] = round(traingroupedMonth.ix[9 + int(i)]['logerror'], 5)
    xgb = XGBRegressor(max_depth=7, learning_rate=0.05, base_score=y_mean, min_child_weight=7, subsample=0.8,
                       colsample_bytree=0.8, n_estimators=200)
    xgb.fit(x_train, y_train)
    pred = xgb.predict(x_test)

    y_pred = []
    for i, predict in enumerate(pred):
        y_pred.append(str(round(predict, 4)))
    #y_pred = np.array(y_pred)
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
