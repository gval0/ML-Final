#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import os
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import dagshub


# In[34]:


# In[36]:

# ### Data Clearing

# In[37]:


class DateParser(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['Date'] = pd.to_datetime(X['Date'])
        return X


# In[38]:


class DataMerger(BaseEstimator, TransformerMixin):
    def __init__(self, stores_df, features_df):
        self.stores_df = stores_df.copy()
        self.features_df = features_df.copy()
        self.features_df['Date'] = pd.to_datetime(self.features_df['Date'])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.merge(self.features_df, on=['Store', 'Date'], how='left')
        X = X.merge(self.stores_df, on='Store', how='left')
        if 'IsHoliday_x' in X.columns:
            X['IsHoliday'] = X['IsHoliday_y'].fillna(X['IsHoliday_x'])
            X = X.drop(['IsHoliday_x', 'IsHoliday_y'], axis=1)

        return X


# In[39]:


class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cleaning_stats_ = {}

    def fit(self, X, y=None):
        if 'Weekly_Sales' in X.columns:
            self.cleaning_stats_['negative_sales_count'] = (X['Weekly_Sales'] < 0).sum()
            self.cleaning_stats_['zero_sales_count'] = (X['Weekly_Sales'] == 0).sum()
        return self

    def transform(self, X):
        X = X.copy()
        if 'Weekly_Sales' in X.columns:
            X['Weekly_Sales'] = X['Weekly_Sales'].abs()

        return X


# ### Feature Engineering

# In[40]:


class TimeFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X['Year'] = X['Date'].dt.year
        X['Month'] = X['Date'].dt.month
        X['Week'] = X['Date'].dt.isocalendar().week
        X['Day'] = X['Date'].dt.day
        X['DayOfWeek'] = X['Date'].dt.dayofweek
        X['DayOfYear'] = X['Date'].dt.dayofyear
        X['Quarter'] = X['Date'].dt.quarter
        X['Month_sin'] = np.sin(2 * np.pi * X['Month'] / 12)
        X['Month_cos'] = np.cos(2 * np.pi * X['Month'] / 12)
        X['Week_sin'] = np.sin(2 * np.pi * X['Week'] / 52)
        X['Week_cos'] = np.cos(2 * np.pi * X['Week'] / 52)
        X['WeeksToChristmas'] = (X['DayOfYear'] - 359).abs()
        X['WeeksToThanksgiving'] = (X['DayOfYear'] - 327).abs()
        X['IsMonthStart'] = (X['Day'] <= 7).astype(int)
        X['IsMonthEnd'] = (X['Day'] >= 24).astype(int)

        return X


# In[41]:


class LagFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, lags=[1, 2, 3, 4, 8, 52], windows=[4, 8, 12]):
        self.lags = lags
        self.windows = windows
        self.historical_data_ = None

    def fit(self, X, y=None):
        if 'Weekly_Sales' in X.columns:
            self.historical_data_ = X[['Store', 'Dept', 'Date', 'Weekly_Sales']].copy()
            self.historical_data_ = self.historical_data_.sort_values(['Store', 'Dept', 'Date'])
        return self

    def transform(self, X):
        X = X.copy()
        X = X.sort_values(['Store', 'Dept', 'Date'])

        if 'Weekly_Sales' in X.columns:
            for lag in self.lags:
                X[f'Sales_Lag_{lag}'] = X.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(lag)

            for window in self.windows:
                X[f'Sales_MA_{window}'] = X.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                )
                X[f'Sales_STD_{window}'] = X.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
                )
        else:
            if self.historical_data_ is not None:
                for lag in self.lags:
                    X[f'Sales_Lag_{lag}'] = np.nan

                for window in self.windows:
                    X[f'Sales_MA_{window}'] = np.nan
                    X[f'Sales_STD_{window}'] = np.nan
        return X


# In[42]:


class StoreDeptFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.store_stats_ = None
        self.dept_stats_ = None
        self.store_dept_stats_ = None

    def fit(self, X, y=None):
        if 'Weekly_Sales' in X.columns:
            self.store_stats_ = X.groupby('Store').agg({
                'Weekly_Sales': ['mean', 'std', 'median'],
                'Size': 'first',
                'Type': 'first'
            })
            self.dept_stats_ = X.groupby('Dept').agg({
                'Weekly_Sales': ['mean', 'std', 'median']
            })
            self.store_dept_stats_ = X.groupby(['Store', 'Dept']).agg({
                'Weekly_Sales': ['mean', 'std', 'count']
            })

        return self

    def transform(self, X):
        X = X.copy()
        X['Is_TypeA'] = (X['Type'] == 'A').astype(int)
        X['Is_TypeB'] = (X['Type'] == 'B').astype(int)
        X['Is_TypeC'] = (X['Type'] == 'C').astype(int)
        X['Size_Bin'] = pd.cut(X['Size'],
                               bins=[0, 50000, 100000, 150000, 300000],
                               labels=['Small', 'Medium', 'Large', 'Extra_Large'])
        if self.store_stats_ is not None:
            store_means = self.store_stats_['Weekly_Sales']['mean'].to_dict()
            store_stds = self.store_stats_['Weekly_Sales']['std'].to_dict()
            X['Store_Avg_Sales'] = X['Store'].map(store_means).fillna(0)
            X['Store_Std_Sales'] = X['Store'].map(store_stds).fillna(0)
            X['Store_CV'] = X['Store_Std_Sales'] / (X['Store_Avg_Sales'] + 1)

            dept_means = self.dept_stats_['Weekly_Sales']['mean'].to_dict()
            dept_stds = self.dept_stats_['Weekly_Sales']['std'].to_dict()
            X['Dept_Avg_Sales'] = X['Dept'].map(dept_means).fillna(0)
            X['Dept_Std_Sales'] = X['Dept'].map(dept_stds).fillna(0)

        return X


# In[43]:


class MarkdownFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.markdown_cols:
            if col in X.columns:
                X[f'{col}_Present'] = (~X[col].isna()).astype(int)
        X['Total_MarkDown'] = X[self.markdown_cols].sum(axis=1, skipna=True)
        X['Active_MarkDowns'] = X[[f'{col}_Present' for col in self.markdown_cols
                                  if f'{col}_Present' in X.columns]].sum(axis=1)
        for col in self.markdown_cols:
            if col in X.columns:
                X[col] = X[col].fillna(0)

        return X


# In[44]:


class EconomicFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.impute_values_ = {}

    def fit(self, X, y=None):
        economic_cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
        for col in economic_cols:
            if col in X.columns:
                self.impute_values_[col] = X[col].median()
        return self

    def transform(self, X):
        X = X.copy()
        for col, value in self.impute_values_.items():
            if col in X.columns:
                X[col] = X[col].fillna(value)

        if 'Temperature' in X.columns:
            X['Temp_Squared'] = X['Temperature'] ** 2
            X['Is_Cold'] = (X['Temperature'] < 32).astype(int)
            X['Is_Hot'] = (X['Temperature'] > 80).astype(int)

        if all(col in X.columns for col in ['Unemployment', 'CPI', 'Fuel_Price']):
            X['Economic_Stress'] = (
                X['Unemployment'] / X['Unemployment'].mean() +
                X['CPI'] / X['CPI'].mean() +
                X['Fuel_Price'] / X['Fuel_Price'].mean()
            ) / 3

        return X


# In[45]:


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders_ = {}
        self.categorical_cols_ = ['Size_Bin']

    def fit(self, X, y=None):
        for col in self.categorical_cols_:
            if col in X.columns:
                le = LabelEncoder()
                # Fit on non-null values
                mask = X[col].notna()
                if mask.any():
                    le.fit(X.loc[mask, col])
                    self.encoders_[col] = le
        return self

    def transform(self, X):
        X = X.copy()

        for col, encoder in self.encoders_.items():
            if col in X.columns:
                X[f'{col}_encoded'] = -1  # Default for unseen/missing
                mask = X[col].notna()
                if mask.any():
                    try:
                        X.loc[mask, f'{col}_encoded'] = encoder.transform(X.loc[mask, col])
                    except ValueError:
                        known_values = encoder.classes_
                        for idx in X[mask].index:
                            if X.loc[idx, col] in known_values:
                                X.loc[idx, f'{col}_encoded'] = encoder.transform([X.loc[idx, col]])[0]

        return X


# In[46]:


class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler_ = RobustScaler()
        self.numeric_features_ = None

    def fit(self, X, y=None):
        exclude = ['Store', 'Dept', 'Date', 'Weekly_Sales']
        self.numeric_features_ = [col for col in X.select_dtypes(include=[np.number]).columns
                                 if col not in exclude]
        if self.numeric_features_:
            self.scaler_.fit(X[self.numeric_features_])

        return self

    def transform(self, X):
        X = X.copy()
        if self.numeric_features_:
            X[self.numeric_features_] = self.scaler_.transform(X[self.numeric_features_])

        return X


# ### Pipeline

# In[47]:


class WalmartPreprocessingPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, stores_df, features_df):
        self.stores_df = stores_df
        self.features_df = features_df

        self.date_parser = DateParser()
        self.data_merger = DataMerger(stores_df, features_df)
        self.data_cleaner = DataCleaner()

        self.time_feature_engineer = TimeFeatureEngineer()
        self.lag_feature_engineer = LagFeatureEngineer()
        self.store_dept_engineer = StoreDeptFeatureEngineer()
        self.markdown_engineer = MarkdownFeatureEngineer()
        self.economic_engineer = EconomicFeatureEngineer()
        self.encoder = CategoricalEncoder()
        self.scaler = FeatureScaler()
        self.is_fitted = False
        self.feature_names_ = None
        self.numeric_features_ = None
        self.categorical_features_ = None

    def fit(self, X, y=None):
        print("Fitting preprocessing pipeline...")
        X_dated = self.date_parser.fit_transform(X)
        X_merged = self.data_merger.fit_transform(X_dated)
        X_cleaned = self.data_cleaner.fit_transform(X_merged)
        X_time = self.time_feature_engineer.fit_transform(X_cleaned)
        X_lag = self.lag_feature_engineer.fit_transform(X_time)
        X_store = self.store_dept_engineer.fit_transform(X_lag)
        X_markdown = self.markdown_engineer.fit_transform(X_store)
        X_economic = self.economic_engineer.fit_transform(X_markdown)

        X_encoded = self.encoder.fit_transform(X_economic)
        X_final = self.scaler.fit_transform(X_encoded)
        self.feature_names_ = [col for col in X_final.columns
                              if col not in ['Date', 'Weekly_Sales']]
        self.numeric_features_ = X_final.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features_ = X_final.select_dtypes(include=['object']).columns.tolist()

        self.is_fitted = True
        print(f"Pipeline fitted. Features created: {len(self.feature_names_)}")
        return self

    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        X_dated = self.date_parser.transform(X)
        X_merged = self.data_merger.transform(X_dated)
        X_cleaned = self.data_cleaner.transform(X_merged)
        X_time = self.time_feature_engineer.transform(X_cleaned)
        X_lag = self.lag_feature_engineer.transform(X_time)
        X_store = self.store_dept_engineer.transform(X_lag)
        X_markdown = self.markdown_engineer.transform(X_store)
        X_economic = self.economic_engineer.transform(X_markdown)
        X_encoded = self.encoder.transform(X_economic)
        X_final = self.scaler.transform(X_encoded)

        return X_final

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# In[48]:


def get_model_ready_data(pipeline_path='preprocessing_pipeline.pkl'):
    pipeline = joblib.load(pipeline_path)
    def preprocess_for_model(raw_data):
        return pipeline.transform(raw_data)
    return preprocess_for_model, pipeline

