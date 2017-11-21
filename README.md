# zillow-house-price-prediction

>* Use five ML models - Neural Network(Keras), Catboost, XGBoost, lightGBM, Adboost 
>* Create more than 100 features (e.g. month log error, neighbor mean price) 
>* Ensemble results of five models by Bagging-Boosting-Stacking 
>* Use data of 4th quarter in 2016 as valid set.  

## 1 Introduction
“Zestimates” are estimated home values based on 7.5 million statistical and machine learning models that analyze hundreds of data points on each property. And, by continually improving the median margin of error (from 14% at the onset to 5% today), Zillow has since become established as one of the largest, most trusted marketplaces for real estate information in the U.S. and a leading example of impactful machine learning.
In this million-dollar competition, participants will develop an algorithm that makes predictions about the future sale prices of homes. 

## 2. Load and check data
### Outlier detection
```python
UP_LIMIT_BODER = 97.5
DOWN_LIMIT_BODER = 2.5
ulimit = np.percentile(train.logerror.values, UP_LIMIT_BODER)
llimit = np.percentile(train.logerror.values, DOWN_LIMIT_BODER)
```

### check for null and missing values
• In Catboost
```python
train_df.fillna(-999, inplace=True)
```
• In NN
```python
imputer = Imputer()
imputer.fit(x_train.iloc[:, :])
x_train = imputer.transform(x_train.iloc[:, :])
```

## Feature engineering
• 3.1 Numerical features

• 3.2 Categorical features

• 3.3 Statistic features
```python
traingroupedMonth = train_df.groupby(["month"])["logerror"].mean().to_frame().reset_index()
traingroupedQuarter = train_df.groupby(["quarter"])["logerror"].mean().to_frame().reset_index()
```
• 3.4 time-series features (can fit the data of 2016 and 2017 well)
```python
df["transaction_year"] = df["transactiondate"].dt.year
df["transaction_month"] = (df["transactiondate"].dt.year - 2016) * 12 + df["transactiondate"].dt.month
df["transaction_quarter"] = (df["transactiondate"].dt.year - 2016) * 4 + df["transactiondate"].dt.quarter
```

## Modeling
• XGBoost
best grade: 0.06442 (without 2017 data)

• lightGBM
best grade: 0.06440 (without 2017 data)

• Neural Network(Keras)
best grade: 0.0646 (without 2017 data)

• Catboost
best grade: 0.06414 (with 2017 data)

• Adaboost
a good choice to reduce overfitting and enhance grade because, in this competition, outlier data can cause a big influence to the final result.

### Model Ensemble
tools: stacknet, mlens
