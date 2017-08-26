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
# drop out ouliers
train_df = train_df[train_df.logerror > -0.4]
train_df = train_df[train_df.logerror < 0.4]
x_train = train_df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print('After removing outliers:')
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

# model
'''
params = [8, 9, 10]
test_scores = []
for param in params:
    xgb = XGBRegressor(max_depth=param, learning_rate=0.05, base_score=y_mean, min_child_weight=1)
    test_score = np.sqrt(-cross_val_score(xgb, x_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
plt.plot(params, test_scores)
plt.title('xgboost error')
plt.show()
'''
xgb = XGBRegressor(max_depth=8, learning_rate=0.05, base_score=y_mean, min_child_weight=1)
xgb.fit(x_train, y_train)
pred = xgb.predict(x_test)

y_pred = []
for i, predict in enumerate(pred):
    y_pred.append(str(round(predict, 4)))
y_pred = np.array(y_pred)

output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
                       '201610': y_pred, '201611': y_pred, '201612': y_pred,
                       '201710': y_pred, '201711': y_pred, '201712': y_pred})
# set col 'ParceID' to first col
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]
output.to_csv(utils.file_output, index=False)

# zip
f = zipfile.ZipFile('output.zip', 'w', zipfile.ZIP_DEFLATED)
f.write('output.csv')
