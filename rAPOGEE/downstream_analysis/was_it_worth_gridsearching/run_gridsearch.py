import pandas as pd
import numpy as np
import pickle as pk
import os, sys
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("../../"), os.pardir)))
from customFunctions.utils import binary_auROC


y = pd.read_csv("../../data/training_test_set_unified.txt", sep="\t", index_col="Mithril_id").label\
    .replace(["benign", "likely benign", "likely pathogenic", "pathogenic"], [0, 0, 1, 1])
X = pd.read_csv("../../checkpoints/mithril_features.csv", sep="\t", index_col=0)
X = X.loc[y.index.intersection(X.index)]
y = y.loc[X.index].astype(int)

embedding = pd.read_csv("../../data/Mithril_embedding_variants_256windows.csv", sep="\t", index_col=0)
embedding.index.name="Mithril_id"
embedding.columns = "channel_"+embedding.columns

X_extended = X.join(embedding)
N_ESTIMATORS = 4096
# N_ESTIMATORS = 64*2


# ==========================================
# PREPROCESSING STEPS
# ==========================================

from sklearn import clone
from scipy import stats
from sklearn.feature_selection import f_classif
from CustomEstimators.FilterByCorrelation import FilterByCorrelation
from CustomEstimators.PartialPCA import PartialPCA
from CustomEstimators.DataFrameUtils import ColumnTransformerDF, SelectFdrDF, SelectFromModelDF, StandardScalerDF
from sklearn.tree import DecisionTreeClassifier

# Feature selection based on importance weights using a balanced Decision Tree
feature_selection = SelectFromModelDF(
    DecisionTreeClassifier(class_weight="balanced", random_state=777), threshold=.01
)

# Feature selection for embedding channels using False Discovery Rate (FDR) with ANOVA F-test
channel_selection = SelectFdrDF(f_classif, alpha=.01)

# Apply parallel feature selection: 'feature_selection' on standard features, 'channel_selection' on embeddings
input_selection = ColumnTransformerDF([
    ("feature_selection", feature_selection, X.columns),
    ("channel_selection", channel_selection, embedding.columns),
], verbose_feature_names_out=False)

# Filter out embedding channels that have a Spearman correlation > 0.25 with standard features to reduce redundancy
channels_correlation_filter = FilterByCorrelation(
    stats.spearmanr,
    features_to_filter=X_extended.columns[(X_extended.columns.str.match("channel_"))], 
    test_features=X_extended.columns[~(X_extended.columns.str.match("channel_"))],
    corr_threshold=0.25
)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScalerDF()

# Apply Principal Component Analysis (PCA) strictly to the embedding channels, reducing them to 5 components
channels_pca = PartialPCA(
    n_components=5, pca_features=embedding.columns
)


# ==========================================
# CLASSIFIER AND HYPERPARAMETER TUNING
# ==========================================

from sklearn.model_selection import ParameterGrid
from CustomEstimators.GridSearchOOB_BinaryClassifier import GridSearchOOB_BinaryClassifier
from sklearn.metrics import roc_auc_score
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline

# Define parameter grid for the Balanced Random Forest
grid_rf = ParameterGrid({
    "criterion": ["gini", "entropy"],
    "max_features": [.1, .25, .5, .75, 1.0],
    "max_depth": [3, 5, 7, 9, 11, 13, 15, 17, 21, 25],
})

# Optimize model using Out-of-Bag (OOB) scores rather than standard CV folds to save time
gridsearch_rf = GridSearchOOB_BinaryClassifier(
    estimator=BalancedRandomForestClassifier(
        n_estimators=N_ESTIMATORS, sampling_strategy={0:22, 1:22}, # Strict undersampling per bootstrap
        oob_score=True, bootstrap=True, n_jobs=-1
    ),
    param_grid=grid_rf, scoring_function=roc_auc_score,
    n_jobs=-1, verbose=99
)

# Assemble the complete Machine Learning Pipeline
model_rf = Pipeline([
    ("input_selection", clone(input_selection)),
    ('channels_correlation_filter', clone(channels_correlation_filter)),
    ("scaler", clone(scaler)),
    ("channels_pca", clone(channels_pca)),
    ("gridsearch", clone(gridsearch_rf)),
])


# ==========================================
# CROSS-VALIDATION EVALUATION
# ==========================================

from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold

# Setup a massive 24-fold stratified cross-validation split
kfold = RepeatedStratifiedKFold(n_splits=24, n_repeats=1, random_state=420)

preprocess_grid = {
    "input_selection__feature_selection__threshold": [.01, .02, .03, .05],
    "input_selection__channel_selection__alpha": [.01, .05, .1],
    "channels_correlation_filter__corr_threshold": [.2, .25, .3, .4],
    "channels_pca__n_components": [1, 2, 3, 5]
}
print("Starting Grid Search with CV...")
print(time.ctime())
grid_search = GridSearchCV(
    model_rf, preprocess_grid, scoring=binary_auROC,
    cv=kfold, n_jobs=1, error_score="raise",
    verbose=99,
).fit(X_extended, y)
print("Grid Search Completed!")
print(time.ctime())
pk.dump(grid_search, open("./gridsearch.pk", "wb"))