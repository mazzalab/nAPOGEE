import pandas as pd
import numpy as np
import warnings
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

class PartialPCA(BaseEstimator, TransformerMixin):

    def __init__(self, pca_features, n_components=None, pc_prefix="PC_", **kwargs):

        self.n_components = n_components
        self.pca_features = pca_features
        self.pc_prefix = pc_prefix
        self.pca = PCA(
            n_components=self.n_components,
            **kwargs
        )

    def fit(self, X, y=None,**fit_params):
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be pandas Dataframe.")
        
        # input features
        self.n_features_in_ = X.shape[1]
        
        # Store the feature names from the DataFrame columns
        self.feature_names_in_ = np.array(X.columns)
        
        # Maschera per le colonne dei channels
        self.feature_mask = X.columns.isin(self.pca_features)
        self.n_features_in_pca = int(self.feature_mask.sum())
        
        # Warning if there are no features to apply PCA on
        if self.n_features_in_pca == 0:
            warnings.warn(f"WARNING: No features to transform.")
            self.n_components_ = 0
            return self

        # Check n_components type
        if type(self.n_components)==int:
            # warning
            if self.n_components > self.n_features_in_pca:
                warnings.warn(f"WARNING: n_components > feature pca features. Setting n_components to {self.n_features_in_pca}.")
                self.n_components = self.n_features_in_pca
                self.pca.n_components = self.n_features_in_pca
            
        elif type(self.n_components)==float:
            if self.n_components <= float(0) or self.n_components > float(1):
                raise ValueError("Invalid explained variance ration value.")
        
        # Ri-setto il numero di pc (nel caso `self.n_components` fosse cambiato)
        self.pca.set_params(n_components=self.n_components)
        # Fitto PCA solo sulle colonne interessate (channels)
        self.pca.fit(X.loc[:, self.feature_mask], y, **fit_params)

        self.n_components_ = self.pca.n_components_

        return self

    
    def transform(self, X, y=None):
        
        # Skipping transformation if there are no features to apply PCA on
        if self.n_features_in_pca == 0:
            warnings.warn(f"WARNING: No features to transform. Returning original X.")
            return X
                
        # Apply mask to X

        if isinstance(X, np.ndarray):
            X_pca = self.pca.transform(X[:, self.feature_mask])
            return np.hstack((X[:, ~self.feature_mask],X_pca))

        elif isinstance(X, pd.DataFrame):
            X_pca = pd.DataFrame(
                self.pca.transform(X.loc[:, self.feature_mask]),
                columns=[f"{self.pc_prefix}{i+1}" for i in range(self.n_components_)],
                index=X.index
            )
            return pd.concat(
                [X.loc[:, ~self.feature_mask], X_pca],
                axis=1
            )
        
        else:
            raise TypeError("expected %s or %s; recieved %s"%(str(np.ndarray), str(pd.DataFrame), str(X.__class__)))

    
    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        return self.fit(X, y).transform(X, y)