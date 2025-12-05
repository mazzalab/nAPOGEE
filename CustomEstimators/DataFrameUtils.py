import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectFdr
from sklearn.compose import ColumnTransformer


class SelectFromModelDF(SelectFromModel):
    def transform(self, X):
        X_new_np = super().transform(X)
        return pd.DataFrame(X_new_np, columns=X.columns[self.get_support()])


class SelectFdrDF(SelectFdr):
    def transform(self, X):
        X_new_np = super().transform(X)
        return pd.DataFrame(X_new_np, columns=X.columns[self.get_support()])


class ColumnTransformerDF(ColumnTransformer):

    def transform(self, X):
        transformed_array = super().transform(X)
        feature_names = self.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed_array, columns=feature_names, index=X.index)
        return transformed_df
    
    def fit_transform(self, X, y=None):
        transformed_array = super().fit_transform(X, y)
        feature_names = self.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed_array, columns=feature_names, index=X.index)
        return transformed_df


class StandardScalerDF(StandardScaler):
    def transform(self, X):
        return pd.DataFrame(super().transform(X), columns=X.columns)
