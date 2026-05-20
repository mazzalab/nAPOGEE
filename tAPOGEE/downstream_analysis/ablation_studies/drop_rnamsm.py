from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.pipeline import Pipeline

import os, sys
import pickle as pk
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("./../../"), os.pardir)))
# from CustomEstimators.PartialPCA import PartialPCA
# from CustomEstimators.FilterByCorrelation import FilterByCorrelation
from CustomEstimators.GridSearchOOB_BinaryClassifier import GridSearchOOB_BinaryClassifier
from CustomEstimators.DataFrameUtils import *


y = pd.read_csv("../../data/training_set.txt", sep="\t", index_col="Mithril_id").label\
    .replace(["benign", "pathogenic"], [0,1])
X = pd.read_csv("../../checkpoints/mithril_features.csv", sep="\t", index_col=0)#.drop(columns=mithril_features)
X = X.loc[y.index.intersection(X.index)]
y = y.loc[X.index].astype(int)


###################################################################################
################################## PREPROCESSING ##################################
###################################################################################

feature_selection = SelectFromModelDF(
    DecisionTreeClassifier(class_weight="balanced", random_state=777), threshold=.01
)
scaler = StandardScalerDF()


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
    ("feature_selection", feature_selection),
    ("scaler", scaler),
    ("gridsearch", gridsearch_svc),
])

preprocess_grid = {
    "feature_selection__threshold": [.01, .02, .03, .05],
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

tuned_model_svc.fit(X, y)
pk.dump(tuned_model_svc, open("./partial_models/model_svc_no_rnamsm.pk", "wb"))

print("################## END ##################")
print("OOB gridsearch params:")
print(tuned_model_svc.best_estimator_[-1].best_params, tuned_model_svc.best_estimator_[-1].best_score)
print("CV gridsearch params:")
print(tuned_model_svc.best_params_, tuned_model_svc.best_score_)