import numpy as np
import warnings
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from joblib import delayed, Parallel

class GridSearchOOB_BinaryClassifier(MetaEstimatorMixin, BaseEstimator):

    def __init__(self, estimator, param_grid, scoring_function,
                 n_jobs=1, refit=True, verbose=0, error_score=np.nan,
                 return_train_score=False):
        
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring_function = scoring_function
        self.n_jobs = n_jobs
        self.refit = refit
        self.verbose = verbose #TODO Mauro: implement verbosity
        self.error_score = error_score
        self.return_train_score = return_train_score


    def _fit_with_params(self, X, y, params_id):
        """
        Fit a deep copy of the estimator on dataset `X`, `y`, using the hyperparameters combination
        at index `params_id` in the `param_grid` oject.
        """

        # clone the estimator in order to create a deep copy of the original one
        estimator = clone(self.estimator)

        # set hyperparameters according to the `param_id` element of `param_grid`
        estimator.set_params(**self.param_grid[params_id])

        #fit the estimator copy
        try:
            estimator.fit(X, y)
            return estimator
        except:
            #if the fitting rises any error, the `None`` is returned
            print("nope")
            return None
        
    
    def _eval_params(self, X, y, params_id):
        """
        Evaluate the hyperparameters combination from `param_grid` at position `params_id`
        on dataset `X`, `y`, according to the `scoring_function`.
        """

        # fit a deep copy of the estimator on dataset `X`, `y`
        # using `self.param_grid[params_id]` hyperparameters
        estimator = self._fit_with_params(X, y, params_id)

        # if the estimator is None it means that some error occours during the fitting
        if estimator is None:
            #return nan for both oob and train score
            return [np.nan, np.nan]
        
        # oob probability prediction for the positive class (1)
        y_oob = estimator.oob_decision_function_[:,1]


        try:
            # try computing the output of `scoring_function` on oob probability
            # this only works if the `scoring_function` can handle continuos prediction (eg AUC)
            oob_score = self.scoring_function(y, y_oob)
        except ValueError:
            # if `scoring_function` rises an error,
            # try to compute the `scoring_function` on oob binary prediction (0, 1)
            # this will be executed if the `scoring_function` can't handle continuos prediction (eg ACC)
            oob_score = self.scoring_function(y, y_oob.round(0).astype(int))

        # if train score is required:
        if self.return_train_score:
            
            # compute the robability prediction on `X` for the positive class (1)
            y_train = estimator.predict_proba(X)[:,1]

            # compute the `scoring_function` for train predictions, as made for oob predictions
            try:
                train_score = self.scoring_function(y, y_oob)
            except ValueError:
                train_score = self.scoring_function(y, y_oob.round(0).astype(int))
            return [oob_score, train_score]
        
        # if train score is not required:
        else:
            # return the oob score and nan as train score
            return [oob_score, np.nan]
    

    def _eval_all_params(self, X, y):
        """
        Evaluate all hyperparameters combinations from `param_grid` on dataset `X`, `y`,
        according the `scoring_function`.
        """
        
        def eval_on_Xy(params_id):
            """
            Wrap  `_eval_params` method, fixing `X` and `y`.
            """
            return self._eval_params(X, y, params_id)
        
        # define a parallel mapping specifying the number of jobs
        parallel = Parallel(n_jobs=self.n_jobs)

        # evaluate each hyperparameters combination in `param_grid`
        # scores is a numpy array of shape (H, 2), where:
        #   H is `len(param_grid)`;
        #   2 is the number of scores for each combinations (one for oob, one for train).
        # note that train score is nan if not required (ie `return_train_score` is `False`)
        scores = np.array(parallel(
            delayed(eval_on_Xy)(params_id)
            for params_id in range(len(self.param_grid))
        ))

        # extract oob scores and train scores
        self.oob_score, self.train_score = scores.T

        # return None
        # TODO Mauro: think about something more useful to return
        return None
    

    @property
    def best_score(self):
        """
        Return the max oob score.
        """
        return np.nanmax(self.oob_score)
    

    @property
    def best_params(self):
        """
        Return the best combination of hyperparameters.
        """
        return self.param_grid[np.nanargmax(self.oob_score)]
    

    @property
    def oob_decision_function_(self):
        """
        Return the oob decision function for each sample in the training set.
        """
        return self.estimator.oob_decision_function_
    

    def fit(self, X, y):
        """
        Perform the greedsearch.
        """

        # compute the oob scores and train scores for each hyperparameters combination
        # note that train score is nan if not required (ie `return_train_score` is `False`)
        self._eval_all_params(X, y)

        # if refit is required
        if self.refit:
            # fit the estimator on the training set with best hyperparameters found
            # note that best hyperparameters combination is that one that maximize the `scoring_function`
            self.estimator.set_params(**self.best_params)
            self.estimator.fit(X, y)
        
        self.classes_ = np.unique(y)
        
        # return the fitted object
        return self
    

    def predict_proba(self, X):
        """
        Useful. Isn't it?
        """
        return self.estimator.predict_proba(X)
    

    def predict(self, X):
        """
        Useful. Isn't it?
        """
        return self.estimator.predict(X)