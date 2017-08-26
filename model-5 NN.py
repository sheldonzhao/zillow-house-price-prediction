import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
import utils
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
import zipfile

print('start reading data')
props = pd.read_csv(utils.file_properties)
train_df = pd.read_csv(utils.file_train)
test_df = pd.read_csv("./input/sample_submission.csv")
test_df = test_df.rename(columns={'ParcelId': 'parcelid'})
train = train_df.merge(props, how='left', on='parcelid')
test = test_df.merge(props, on='parcelid', how='left')

for c in train.columns:
    if train[c].dtype == 'float64':
        train[c] = train[c].values.astype('float32')

print("Done with Merged Operation")
#####Removing Outliers, Total Features#####
UP_LIMIT_BODER = 98
DOWN_LIMIT_BODER = 2
ulimit = np.percentile(train.logerror.values, UP_LIMIT_BODER)
llimit = np.percentile(train.logerror.values, DOWN_LIMIT_BODER)
print('the logerror = %f < %f percent' % (ulimit, UP_LIMIT_BODER))
print('the logerror = %f > %f percent' % (llimit, DOWN_LIMIT_BODER))
train_df = train_df[train_df.logerror >= llimit]
train_df = train_df[train_df.logerror <= ulimit]

do_not_include = ['parcelid', 'logerror', 'transactiondate']
feature_names = [f for f in train.columns if f not in do_not_include]
print("We have %i features." % len(feature_names))

#####Getting the same number of columns for Train, Test######
y = train['logerror'].values
train = train[feature_names].copy()
test = test[feature_names].copy()

#####Handling Missing Values#####

for i in range(len(train.columns)):
    train.iloc[:, i] = (train.iloc[:, i]).fillna(0)

for i in range(len(test.columns)):
    test.iloc[:, i] = (test.iloc[:, i]).fillna(0)

#####Encoding the Categorical Variables#####

lbl = LabelEncoder()
for c in train.columns:
    if train[c].dtype == 'object':
        lbl.fit(list(train[c].values))
        train[c] = lbl.transform(list(train[c].values))

for c in test.columns:
    if test[c].dtype == 'object':
        lbl.fit(list(test[c].values))
        test[c] = lbl.transform(list(test[c].values))


for i in range(len(train.columns)):
    train.iloc[:, i] = (train.iloc[:, i]).astype(float)

for i in range(len(test.columns)):
    test.iloc[:, i] = (test.iloc[:, i]).astype(float)

print("Done with the Encoding")
####Normalizing the values####

mmScale = MinMaxScaler()
n = train.shape[1]
x_train = mmScale.fit_transform(train)
x_test = mmScale.fit_transform(test)


#####Artificial Neural Networks Implementation#####
print("Starting Neural Network")
model_n = Sequential()
# Want to use an expotential linear unit instead of the usual relu
model_n.add(Dense(n, activation='relu', input_shape=(n,)))
model_n.add(Dense(int(0.5 * n), activation='relu'))
model_n.add(Dense(int(0.3 * n), activation='relu'))
model_n.add(Dense(int(0.1 * n), activation='relu'))
model_n.add(Dense(1, activation='linear'))
model_n.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
model_n.fit(x_train, y, epochs=5, batch_size=10)
predict_test_NN = model_n.predict(x_test)
print(predict_test_NN)
#####Writing the Results######
sub = pd.read_csv('./input/sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = predict_test_NN

print('Writing csv ...')
sub.to_csv(utils.file_output, index=False, float_format='%.4f')
# zip
f = zipfile.ZipFile('output.zip', 'w', zipfile.ZIP_DEFLATED)
f.write('output.csv')
