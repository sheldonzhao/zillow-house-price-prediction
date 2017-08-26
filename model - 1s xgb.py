import pandas as pd
import utils
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import zipfile
import operator
from sklearn.model_selection import GridSearchCV

train_df = pd.read_csv(utils.file_merge_data)
parcelid = train_df['parcelid']
# 1. filter float features
print('start filtering float features')
table_type = train_df.dtypes.reset_index()
table_type.columns = ["feature", "Type"]
# print(table_type.groupby("Type").aggregate('count'))
float_feature_set = table_type[table_type['Type'] == 'float64']['feature']
print('the number of float features is %d', len(float_feature_set))
# print(float_feature_set)
train_df = train_df[float_feature_set]

# 2. preprocess
print('start preprocess')
for c, dtype in zip(train_df.columns, train_df.dtypes):
    if dtype == np.float64:
        train_df[c] = train_df[c].astype(np.float16)

for feat in train_df.columns:
    if 'id' in feat:
        print(feat)
        feat_df = pd.get_dummies(train_df[feat], prefix=feat)
        train_df = pd.concat([train_df, feat_df], axis=1)
        train_df.pop(feat)

# 3. split training set ans test set
test_set = train_df.copy()
train_set = train_df[train_df['logerror'].notnull()]

y_label = train_set.pop('logerror')
test_set.pop('logerror')
print('the number of features set is %d', len(train_set.columns))

# 4. xgb
print('start modeling')
X_train, X_test, y_train, y_test = train_test_split(train_set, y_label, test_size=0, random_state=0)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
# test_set = xgb.DMatrix(test_set)

parameters = {
    'learning_rate': [0.1],  # so called `eta` value
    'max_depth': [3, 4, 5],
    'min_child_weight': [3, 4, 5],
    'silent': [1],
    'subsample': [0.5, 0.8],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'n_estimators': [500, 1000],  # number of trees, change it to 1000 for better results
    # 'gamma': [0]
    # 'eval_metric':logloss
}
xgb_model = xgb.XGBRegressor()
bst = GridSearchCV(xgb_model, parameters, cv=5, verbose=3)

bst.fit(X_train, y_train)
y = bst.predict(test_set)
print(bst.best_score_)
print(bst.best_params_)

# output
print('start generating result')
res = pd.DataFrame(
    {'201610': y, '201611': y, '201612': y, '201710': y, '201711': y,
     '201712': y})
res = pd.concat([parcelid, res], axis=1)
res.to_csv(utils.file_output, index=False)
# zip
f = zipfile.ZipFile('output.zip', 'w', zipfile.ZIP_DEFLATED)
f.write('output.csv')

'''
# feature importance
feature_score = bst.get_fscore()
feature_score = sorted(feature_score.items(), key=operator.itemgetter(1))
print(feature_score)
df = pd.DataFrame(feature_score, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(16, 10))
plt.show()

'''
