import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FilterByCorrelation(BaseEstimator, TransformerMixin):

    def __init__(self, correlation_method, corr_threshold, features_to_filter, test_features, **kwargs):
        """
        Initializes the correlation filter.

        Parameters:
        - correlation_method: callable, correlation method (e.g., scipy.stats.spearmanr)
        - corr_threshold: float, correlation threshold for filtering
        - features_to_filter: list or array-like, features to potentially filter out
        - test_features: list or array-like, features to test against
        - kwargs: additional keyword arguments for the correlation method
        """
        self.correlation_method = correlation_method
        self.features_to_filter = features_to_filter
        self.test_features = test_features
        self.corr_threshold = corr_threshold
        self.corr_kwargs = kwargs


        # Check intersection of the 2 feature list
        if len (set(self.test_features) & set(self.features_to_filter)) > 0:
            raise ValueError(
                'There are repeated features: %s'%str(set(self.test_features) & set(self.features_to_filter))
            )
    

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fits the correlation filter to the data.

        Parameters:
        - X: pd.DataFrame of shape (n_samples, n_features)
        - y: Ignored, present for API consistency

        Returns:
        self
        """
        
        # Check if X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError('%s fit only supports Dataframe input'%self.__class__.__name__)
        
        # Store the number of features in the input data
        self.n_features_in_ = X.shape[1]

        # Store the feature names from the DataFrame columns
        self.feature_names_in_ = np.array(X.columns)

        # Create a boolean mask to identify columns to be filtered
        self.features_to_filter_mask = X.columns.isin(self.features_to_filter)

        # Create a boolean mask to identify columns to be used for testing correlation
        self.test_features_mask = X.columns.isin(self.test_features)

        # Initialize a matrix to store the correlation scores between features
        self.corr_matrix = np.zeros((self.features_to_filter_mask.sum() , self.test_features_mask.sum()))

        # Iterate over each feature to be filtered
        for i, col1 in enumerate(X.columns[self.features_to_filter_mask]):
            # For each feature to be filtered, iterate over each test feature
            for j, col2 in enumerate(X.columns[self.test_features_mask]):
                # Compute the correlation score between the feature to filter and the test feature
                score, pval = self.correlation_method(X[col1], X[col2], **self.corr_kwargs)
                # Store the correlation score in the correlation matrix
                self.corr_matrix[i,j] =  score
                
        return self


    def get_support(self) -> np.ndarray:
        """
        Get a mask, or integer index, of the features selected.

        Returns:
        np.ndarray: Boolean mask of selected features
        """

        # Determine which features to filter based on the correlation threshold
        # This checks if the absolute value of any correlation score in the matrix
        # exceeds the threshold for each feature to filter.
        filtered_features = (np.abs(self.corr_matrix) > self.corr_threshold).any(axis=1)
        
        # Initialize a boolean array to keep all features initially
        feat_to_take = np.ones(self.n_features_in_, dtype=bool)

        # Update the mask to filter out features that exceed the correlation threshold
        # For the features to filter, set their corresponding mask values to False
        feat_to_take[self.features_to_filter_mask] = ~filtered_features        

        return feat_to_take

    def transform(self, X, y=None):
            """
            Reduces the data to the selected features.

            Parameters:
            - X: ndarray or pd.DataFrame of shape (n_samples, n_features)
            - y: Ignored, present for API consistency
            
            Returns:
            The input data with only the selected features.
            """
            if isinstance(X, np.ndarray):
                X_transformed = X[:, self.get_support()]
            elif isinstance(X, pd.DataFrame):
                X_transformed = X.iloc[:, self.get_support()]
            else:
                raise TypeError(
                    "expected %s or %s; recieved %s"%(str(np.ndarray), str(pd.DataFrame), str(X.__class__))
                )
    
            return X_transformed
        
    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Parameters:
        - X: ndarray or pd.DataFrame of shape (n_samples, n_features)
        - y: Ignored, present for API consistency

        Returns:
        pd.DataFrame: The transformed data.
        """
        return self.fit(X, y).transform(X)