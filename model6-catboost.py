import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from catboost import CatBoostRegressor
from tqdm import tqdm
import utils

print('Loading Properties ...')
properties2016 = pd.read_csv('./input/properties_2016.csv', low_memory=False)
properties2017 = pd.read_csv('./input/properties_2017.csv', low_memory=False)

print('Loading Train ...')
train2016 = pd.read_csv('./input/train_2016_v2.csv', parse_dates=['transactiondate'], low_memory=False)
train2017 = pd.read_csv('./input/train_2017.csv', parse_dates=['transactiondate'], low_memory=False)


def add_date_features(df):
    df["transaction_year"] = df["transactiondate"].dt.year
    df["transaction_month"] = (df["transactiondate"].dt.year - 2016) * 12 + df["transactiondate"].dt.month
    df["transaction_quarter"] = (df["transactiondate"].dt.year - 2016) * 4 + df["transactiondate"].dt.quarter
    df.drop(["transactiondate"], inplace=True, axis=1)
    return df


train2016 = add_date_features(train2016)
train2017 = add_date_features(train2017)

print('Loading Sample ...')
sample_submission = pd.read_csv('./input/sample_submission.csv', low_memory=False)

print('Merge Train with Properties ...')
train2016 = pd.merge(train2016, properties2016, how='left', on='parcelid')
train2017 = pd.merge(train2017, properties2017, how='left', on='parcelid')

print('Tax Features 2017  ...')
train2017.iloc[:, train2017.columns.str.startswith('tax')] = np.nan

print('Concat Train 2016 & 2017 ...')
train_df = pd.concat([train2016, train2017], axis=0)
test_df = pd.merge(sample_submission[['ParcelId']], properties2016.rename(columns={'parcelid': 'ParcelId'}), how='left',
                   on='ParcelId')

# statistic
traingroupedYear = train_df.groupby(["transaction_year"])["logerror"].mean().to_frame().reset_index()
traingroupedMonth = train_df.groupby(["transaction_month"])["logerror"].mean().to_frame().reset_index()
traingroupedQuarter = train_df.groupby(["transaction_quarter"])["logerror"].mean().to_frame().reset_index()

train_df['year_logerror'] = train_df['transaction_year'].map(
    lambda x: round(traingroupedYear.ix[int(x) - 2016]['logerror'], 6))
train_df['month_logerror'] = train_df['transaction_month'].map(
    lambda x: round(traingroupedMonth.ix[int(x) - 1]['logerror'], 6))
train_df['quarter_logerror'] = train_df['transaction_quarter'].map(
    lambda x: round(traingroupedQuarter.ix[int(x) - 1]['logerror'], 6))

print("Define training features !!")
exclude_other = ['parcelid', 'logerror', 'propertyzoningdesc']
train_features = []
for c in train_df.columns:
    if c not in exclude_other:
        train_features.append(c)
print("We use these for training: %s" % len(train_features))

print("Define categorial features !!")
cat_feature_inds = []
cat_unique_thresh = 1000
for i, c in enumerate(train_features):
    num_uniques = len(train_df[c].unique())
    if num_uniques < cat_unique_thresh \
            and not 'sqft' in c \
            and not 'cnt' in c \
            and not 'nbr' in c \
            and not 'number' in c \
            and not 'logerror' in c:
        cat_feature_inds.append(i)
print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])

print("Replacing NaN values by -999 !!")
train_df.fillna(-999, inplace=True)
test_df.fillna(-999, inplace=True)

print("Training time !!")
X_train = train_df[train_features]
y_train = train_df.logerror
print(X_train.shape, y_train.shape)

num_ensembles = 5
res = []

for i in range(3):
    print('running %d' % i)
    y_pred = 0.0
    test_df['transaction_year'] = 2016
    test_df['transaction_month'] = 10 + i
    test_df['transaction_quarter'] = 4
    test_df['year_logerror'] = test_df['transaction_year'].map(
        lambda x: round(traingroupedYear.ix[int(x) - 2016]['logerror'], 6))
    test_df['month_logerror'] = test_df['transaction_month'].map(
        lambda x: round(traingroupedMonth.ix[int(x) - 1]['logerror'], 6))
    test_df['quarter_logerror'] = test_df['transaction_quarter'].map(
        lambda x: round(traingroupedQuarter.ix[int(x) - 1]['logerror'], 6))

    X_test = test_df[train_features]
    for j in tqdm(range(num_ensembles)):
        print('running inner %d' % j)
        model = CatBoostRegressor(
            iterations=630, learning_rate=0.03,
            depth=6, l2_leaf_reg=3,
            loss_function='MAE',
            eval_metric='MAE',
            random_seed=j)
        model.fit(
            X_train, y_train,
            cat_features=cat_feature_inds)
        y_pred += model.predict(X_test)
    y_pred /= num_ensembles
    res.append(y_pred)

output = pd.DataFrame({'ParcelId': test_df['ParcelId'].astype(np.int32),
                       '201610': res[0], '201611': res[1], '201612': res[2],
                       '201710': res[0], '201711': res[1], '201712': res[2]})
# set col 'ParceID' to first col
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]
output.to_csv(utils.file_output1, float_format='%.6f', index=False)
print('reaching to an end')
