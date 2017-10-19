import pandas as pd
import zipfile

file_merge_data = './input/merge_data.csv'
file_properties = './input/properties_2016.csv'
file_properties_2017 = './input/properties_2017.csv'
file_train = './input/train_2016_v2.csv'
file_train_2017 = './input/train_2017.csv'
file_output = './output.csv'
file_test = './test.csv'
file_output_log = './output-log.csv'
file_output_xgb = './output-xgb.csv'
file_output_regressor = './output-regressor.csv'
file_2430 = './1.csv'
file_1328 = './output 2.csv'


def print_len(data):
    return len(data['parcelid'])


def print_feature_unique_value():
    data = pd.read_csv(file_merge_data)
    print('the number of features is %d' % len(data.columns))
    for i in data.columns:
        print('the feature is %s' % i)
        print(data[i].unique())


def predict_feat(properties, feat):
    pass


def get_miss_ratio(data, threshold):
    miss_ratio = data.isnull().sum() / data.shape[0]
    res = []
    for feat, ratio in miss_ratio.iteritems():
        if ratio < threshold:
            res.append(feat)
    return res

