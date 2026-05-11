###################################################################################
##################################### IMPORT ######################################
###################################################################################

import pandas as pd
import numpy as np
import pickle as pk
from scipy import stats

from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.pipeline import Pipeline

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("./../../"), os.pardir)))
from CustomEstimators.PartialPCA import PartialPCA
from CustomEstimators.FilterByCorrelation import FilterByCorrelation
from CustomEstimators.GridSearchOOB_BinaryClassifier import GridSearchOOB_BinaryClassifier
from CustomEstimators.DataFrameUtils import *

positional_features = ['Human_MSA_position', 'closest_modification', 'closest_modification_2D']

###################################################################################
##################################### DATASET #####################################
###################################################################################

y = pd.read_csv("../../data/training_set.txt", sep="\t", index_col="Mithril_id").label\
    .replace(["benign", "pathogenic"], [0,1])
X = pd.read_csv("../../checkpoints/mithril_features.csv", sep="\t", index_col=0).drop(columns=positional_features)
X = X.loc[y.index.intersection(X.index)]
y = y.loc[X.index].astype(int)


embedding = pd.read_csv("../../data/Mithril_msm_embedding.csv", index_col=0)
embedding.index.name = "Mithril_id"
embedding.columns = "channel_"+embedding.columns.astype(str)

X_extended = X.join(embedding)


###################################################################################
################################## PREPROCESSING ##################################
###################################################################################

feature_selection = SelectFromModelDF(
    DecisionTreeClassifier(class_weight="balanced", random_state=777), threshold=.01
)

channel_selection = SelectFdrDF(f_classif, alpha=.005)

input_selection = ColumnTransformerDF([
    ("feature_selection", feature_selection, X.columns),
    ("channel_selection", channel_selection, embedding.columns),
], verbose_feature_names_out=False)

channels_correlation_filter = FilterByCorrelation(
    stats.spearmanr,
    features_to_filter=X_extended.columns[(X_extended.columns.str.match("channel_"))], 
    test_features=X_extended.columns[~(X_extended.columns.str.match("channel_"))],
    corr_threshold=0.25
)

scaler = StandardScalerDF()

channels_pca = PartialPCA(
    n_components=10, pca_features=embedding.columns
)


###################################################################################
################################### CLASSIFIER ####################################
###################################################################################

grid_svc = ParameterGrid({
    "estimator__C": [0.1, 1, 10, 100],
    "estimator__gamma": ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1.],
    "max_features": [.1, .25, .5, .75, 1.],
})

gridsearch_svc = GridSearchOOB_BinaryClassifier(
    estimator=BalancedBaggingClassifier(
        estimator=SVC(kernel="rbf", probability=True),
        n_estimators=500, oob_score=True, replacement=True, bootstrap=True,
        sampling_strategy="not minority", n_jobs=8
    ),
    param_grid=grid_svc,
    scoring_function=roc_auc_score,
    n_jobs=8
)

model_svc = Pipeline([
    ("input_selection", input_selection),
    ('channels_correlation_filter', channels_correlation_filter),
    ("scaler", scaler),
    ("channels_pca", channels_pca),
    ("gridsearch", gridsearch_svc),
])

preprocess_grid = {
    "input_selection__feature_selection__threshold": [.01, .02, .03, .05],
    "input_selection__channel_selection__alpha": [.01, .05, .1],
    "channels_correlation_filter__corr_threshold": [.2, .25, .3, .4],
    "channels_pca__n_components": [1, 2, 3, 5]
}

k_fold_gridsearch = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
def binary_auROC(clf, input_data, binary_target):
    return roc_auc_score(binary_target, clf.predict_proba(input_data)[:,1])

tuned_model_svc = GridSearchCV(
    estimator=model_svc, param_grid=preprocess_grid,
    cv=k_fold_gridsearch, error_score=np.nan, refit=True,
    n_jobs=1, verbose=2,
    scoring=binary_auROC,
)


###################################################################################
################################ TUNING / TRAINING ################################
###################################################################################

tuned_model_svc.fit(X_extended, y)
pk.dump(tuned_model_svc, open("./partial_models/model_svc_no_position.pk", "wb"))

print("################## END ##################")
print("OOB gridsearch params:")
print(tuned_model_svc.best_estimator_[-1].best_params, tuned_model_svc.best_estimator_[-1].best_score)
print("CV gridsearch params:")
print(tuned_model_svc.best_params_, tuned_model_svc.best_score_)
