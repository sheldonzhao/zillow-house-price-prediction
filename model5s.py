import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import utils
import zipfile
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler

print('start reading data')
properties = pd.read_csv(utils.file_properties)
train = pd.read_csv(utils.file_train)

print('start preprocessing')
for c, dtype in zip(properties.columns, properties.dtypes):
    if dtype == np.float64:
        properties[c] = properties[c].astype(np.float32)

for c in properties.columns:
    properties[c] = properties[c].fillna(0)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

train_df = train.merge(properties, how='left', on='parcelid')
x_train = train_df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
x_test = properties.drop(['parcelid'], axis=1)
# shape
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

# drop out ouliers
UP_LIMIT_BODER = 98
DOWN_LIMIT_BODER = 2
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

print("Starting Neural Network")
mmScale = MinMaxScaler()
x_train = mmScale.fit_transform(x_train)
x_test = mmScale.fit_transform(x_test)

model_n = Sequential()
n = x_train.shape[1]
# Want to use an expotential linear unit instead of the usual relu
model_n.add(Dense(n, activation='relu', input_shape=(n,)))
model_n.add(Dense(int(0.8 * n), activation='relu'))
model_n.add(Dense(int(0.5 * n), activation='relu'))
model_n.add(Dense(int(0.3 * n), activation='relu'))
model_n.add(Dense(int(0.1 * n), activation='relu'))
model_n.add(Dense(1, activation='linear'))
model_n.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
model_n.fit(x_train, y_train, epochs=20, batch_size=10)
pred = model_n.predict(x_test)
print(pred)
print(type(pred))

output = pd.read_csv('./input/sample_submission.csv')
for c in output.columns[output.columns != 'ParcelId']:
    output[c] = pred

print('Writing csv ...')
output.to_csv(utils.file_output, index=False, float_format='%.4f')
# zip
f = zipfile.ZipFile('output.zip', 'w', zipfile.ZIP_DEFLATED)
f.write('output.csv')

