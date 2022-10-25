import numpy as np
import math
from ._kd_tree import KDTree

from ._utils import mc_sampling,aknn




class NNDE(object):
    def __init__(
        self,
        metric="euclidean",
        leaf_size=40,
        seed=1,
        score_criterion="MISE",
        sampling_stratigy="bounded",
        max_neighbors="auto",
        score_validate_scale="auto"
    ):
        self.metric = metric
        self.leaf_size=leaf_size
        self.seed=seed
        self.score_criterion=score_criterion
        self.sampling_stratigy=sampling_stratigy
        self.max_neighbors=max_neighbors
        self.score_validate_scale=score_validate_scale
        
        
        if metric not in KDTree.valid_metrics:
            raise ValueError("invalid metric: '{0}'".format(metric))
        
        self.log_density=None
        
        
        

    def fit(self, X, y=None):
       
        if self.max_neighbors=="auto":
            self.max_neighbors_=min(int(X.shape[0]*(2/3)),10000)
        
            
        if self.score_validate_scale=="auto":
            self.score_validate_scale_=self.n_train_*(self.dim_*2)
        

        self.tree_ = KDTree(
            X,
            metric=self.metric,
            leaf_size=self.leaf_size,
        )
        
        self.dim_=X.shape[1]
        self.n_train_=X.shape[0]
        self.vol_unitball_=math.pi**(self.dim_/2)/math.gamma(self.dim_/2+1)
        
        
        return self
    
    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in ['k','threshold_r','threshold_num','C']:
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out
    
    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)


        for key, value in params.items():
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))
            setattr(self, key, value)
            valid_params[key] = value

        return self
    
    def score_samples(self, X):
        pass
    
    
    def predict(self,X,y=None):

        
        self.log_density=self.score_samples(X)
        
        return self.log_density
    
    
    def compute_KL(self,X):

        
        # Monte Carlo estimation of integral
        kwargs={"ruleout":0.01,"method":self.sampling_stratigy,"seed":self.seed}
        X_validate,pdf_X_validate=mc_sampling(X,nsample=self.score_validate_scale_,**kwargs)
        validate_log_density=self.score_samples(X_validate)
        
        # if density has been computed, then do not update object attribute
        if self.log_density is None:
            self.log_density=self.score_samples(X)
        
        return self.log_density.mean()-(np.exp(validate_log_density)/pdf_X_validate).mean()
    
        
    def compute_MISE(self,X):

        
        # Monte Carlo estimation of integral
        kwargs={"ruleout":0.01,"method":self.sampling_stratigy,"seed":self.seed}
        X_validate,pdf_X_validate=mc_sampling(X,nsample=self.score_validate_scale_,**kwargs)
        validate_log_density=self.score_samples(X_validate)
        
        # if density has been computed, then do not update object attribute
        if self.log_density is None:
            self.log_density=self.score_samples(X)
        
        return 2*np.exp(self.log_density).mean()-(np.exp(2*validate_log_density)/pdf_X_validate).mean()
    
    def compute_ANLL(self,X):

        
        # if density has been computed, then do not update object attribute
        if self.log_density is None:
            self.log_density=self.score_samples(X)

        return -self.log_density.mean()

    def score(self, X, y=None):

        self.ANLL=self.compute_ANLL(X)
        
        if self.score_criterion=="KL":
            self.KL=self.compute_KL(X)
            return self.KL
        elif self.score_criterion=="MISE":
            self.MISE=self.compute_MISE(X)
            return self.MISE
    
    
    
    
class AKNN(NNDE):
    def __init__(self,
                 beta=1,
                 C=1,
                 cut_off=5,
                 save_weights=False,
                 threshold_num=5,
                 threshold_r=0.5,
                 k=2):
        super(AKNN, self).__init__()
        self.beta=beta
        self.C=C

    def score_samples(self, X):
        
        
        log_density=aknn(X,self.tree_,self.k,self.n_train_,self.dim_,
                        self.vol_unitball_,self.kmax,self.C,self.beta)

        return log_density






    