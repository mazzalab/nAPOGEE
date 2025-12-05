import warnings
import numpy as np
from scipy import stats


class ZeroOneInflatedBeta():

    def __init__(self, discrete_p=None, bernoulli_p=None, beta_a=None, beta_b=None):

        if not discrete_p is None:
            self.switch_model = stats.bernoulli(p=discrete_p)
        if not bernoulli_p is None:
            self.bernoulli_model = stats.bernoulli(p=bernoulli_p)
        if (not beta_a is None) and  (not beta_b is None):
            self.beta_model = stats.beta(beta_a, beta_b)
        elif (not beta_a is None) or (not beta_b is None):
            raise ValueError("You must define both `beta_a` and `beta_b`, not just one of them.")

        # TODO implement a robust parameter error estimation
        self.discrete_p_error = None
        self.bernoulli_p_error = None
        self.beta_a_error = None
        self.beta_b_error = None
    
    def _fit_switch_model(self, x):
        # x must be a numpy array
        self.switch_model = stats.bernoulli(
            p = np.logical_or(x==0, x==1).mean() # P(x in {0, 1})
        )
        # TODO estimate the error in the probability estimation
    
    def _fit_bernoulli_model(self, x):
        # x must be a numpy array
        zeros = (x==0).sum()
        ones = (x==1).sum()
        if zeros+ones<1:
            warnings.warn("No extremes values have been found. Bernoulli model will not be fitted.")
            return
        self.bernoulli_model = stats.bernoulli(
            p = ones/(ones+zeros) # P(x=1 | x in {0, 1})
        )
        # TODO estimate the error in the probability estimation
    
    def _fit_beta_model(self, x):
        # x must be a numpy array
        # Approximate P(x | 0<x<1) to a beta distribution
        if np.logical_and(x>0, x<1).sum()<1:
            warnings.warn("No intermediate values have been found. Beta model will not be fitted.")
            return
        self.beta_model = stats.beta(
            *stats.beta.fit(
                x[np.logical_and(x>0, x<1)],
                floc=0, fscale=1
            )
        )
        # TODO estimate the error in a and b estimation
    
    def fit(self, x):
        x = np.array(x)
        self._fit_switch_model(x)
        self._fit_bernoulli_model(x)
        self._fit_beta_model(x)
    
    
    def pdf(self, x):

        x = np.array(x)
        discrete_ids = np.logical_or(x==0, x==1)
        
        try:
            bernoulli_pmf = self.bernoulli_model.pmf(x[discrete_ids])*self.switch_model.mean()
        except AttributeError:
            bernoulli_pmf = np.zeros_like(x[discrete_ids])
        
        try:
            beta_pdf = self.beta_model.pdf(x[~discrete_ids]) * (1-self.switch_model.mean())
        except AttributeError:
            beta_pdf = np.zeros_like(x[~discrete_ids])
        
        pdf_values = np.zeros_like(x)
        pdf_values[discrete_ids] = bernoulli_pmf
        pdf_values[~discrete_ids] = beta_pdf
        
        return pdf_values
    

    def __repr__(self):
        
        repr_out = ""
        
        try:
            repr_out+="%.2f"%(self.switch_model.mean())
        except AttributeError:
            return super().__repr__()
        
        try:
            repr_out+= " * Bernoulli(%.2f)"%(self.bernoulli_model.mean())
        except AttributeError:
            repr_out+= " * Bernoulli(?)"
        
        repr_out+=" + %.2f"%(1-self.switch_model.mean())

        try:
            repr_out+= " * Beta(%.2f, %.2f)"%(self.beta_model.args[0], self.beta_model.args[1])
        except AttributeError:
            repr_out+= " * Beta(?)"
        
        return repr_out