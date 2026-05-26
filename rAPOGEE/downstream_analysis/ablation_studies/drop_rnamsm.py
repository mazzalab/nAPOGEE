import pandas as pd
import numpy as np
import pickle as pk
import os, sys
from sklearn.metrics import roc_curve, average_precision_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("../../"), os.pardir)))
from customFunctions.utils import binary_auROC

y = pd.read_csv("../../data/training_test_set_unified.txt", sep="\t", index_col="Mithril_id").label\
    .replace(["benign", "likely benign", "likely pathogenic", "pathogenic"], [0, 0, 1, 1])
# Load genomic features and drop the conservation features defined above
X = pd.read_csv("../../checkpoints/mithril_features.csv", sep="\t", index_col=0).drop("MLC_score", axis=1)
X = X.loc[y.index.intersection(X.index)]
y = y.loc[X.index].astype(int)

N_ESTIMATORS = 4096


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

feature_selection = SelectFromModelDF(
    DecisionTreeClassifier(class_weight="balanced", random_state=777), threshold=.01
)
scaler = StandardScalerDF()


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
    n_jobs=16, verbose=99
)

# Assemble the complete Machine Learning Pipeline
model_rf = Pipeline([
    ("feature_selection", clone(feature_selection)),
    ("scaler", clone(scaler)),
    ("gridsearch", clone(gridsearch_rf)),
])


# ==========================================
# CROSS-VALIDATION EVALUATION
# ==========================================

from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold

# Setup a massive 24-fold stratified cross-validation split
kfold = RepeatedStratifiedKFold(n_splits=24, n_repeats=1, random_state=420)

# Run cross-validation evaluating performance via a custom binary AUROC metric
cross_validation_result = cross_validate(
    model_rf, X, y, scoring=binary_auROC,
    cv=kfold, return_estimator=True,
    n_jobs=1, error_score="raise", verbose=99,
)


cv_predictions = [
    pipe.predict_proba(X.iloc[test])[:,1]
    for (train, test), pipe in zip(kfold.split(X, y), cross_validation_result["estimator"])
]
cv_ground_truth = [
    y.iloc[test]
    for train, test in kfold.split(X, y)
]
pk.dump((cv_ground_truth, cv_predictions), open("./ablation_cv/cv_predictions_no_rnamsm.pk", "wb"))