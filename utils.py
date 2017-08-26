import pandas as pd

file_merge_data = './input/merge_data.csv'
file_properties = './input/properties_2016.csv'
file_train = './input/train_2016_v2.csv'
file_output = './output.csv'
file_test = './test.csv'


def print_len(data):
    return len(data['parcelid'])


def print_feature_unique_value():
    data = pd.read_csv(file_merge_data)
    print('the number of features is %d' % len(data.columns))
    for i in data.columns:
        print('the feature is %s' % i)
        print(data[i].unique())
