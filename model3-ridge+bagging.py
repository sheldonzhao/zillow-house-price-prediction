import numpy as np
import pandas as pd
import xgboost as xgb
import utils
import zipfile
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from sklearn.ensemble import BaggingRegressor


properties = pd.read_csv(utils.file_properties)
train = pd.read_csv(utils.file_train)

print('start to preprocess')
for c in properties.columns:
    properties[c] = properties[c].fillna(0)
    if properties[c].dtype == 'object':
        properties.pop(c)

# create training set and test set
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

# Ridge
clf = BaggingRegressor(n_estimators=100, base_estimator=Ridge(15), max_features=0.8)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)

'''
# bagging
params = [10,15,20,30,40,50,100]
test_scores = []
for param in params:
    clf = BaggingRegressor(n_estimators=param, base_estimator=Ridge(15), max_features=0.8)
    test_score = np.sqrt(-cross_val_score(clf, x_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
plt.plot(params, test_scores)
plt.show()
plt.title('Bagging error')


# alphas = np.logspace(-3, 2, 50)
alphas=range(0, 20, 1)
test_scores = []
for alpha in alphas:
    clf = Ridge(alpha)
    test_score = np.sqrt(-cross_val_score(clf, x_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
plt.plot(alphas, test_scores)
plt.show()
plt.title('Alpha vs CV Error')
'''

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
