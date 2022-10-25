'''
Utility Functions
-----------------
'''

import numpy as np

from ._distributions import MultivariateNormalDistribution,TDistribution,MixedDistribution




def mc_sampling(X,nsample,**kwargs):
    """Monte Carlo Sampling. 
    Generate importance sampling points and report their likelihood.

    Parameters
    ----------
    X : array-like of shape (n_train, dim_)
        List of n_train-dimensional data points.  Each row
        corresponds to a single data point.
        
    nsample : int
        Number of instances to generate. 
        
    Args:
        **method : {"bounded", "heavy_tail", "normal", "mixed"}
            Importance sampling methods to choose. Use "bounded" if all 
            entries are bounded. Use "normal" if data is concentrated. Use 
            "heavy_tail" or "mixed" if data is heavy tailed but pay attention 
            to numerical instability.
        **ruleout : float 
            Quantile for ruling out certain range of outliers. 
     
    Returns
    -------
    X_validate : array-like of shape (nsample, dim_)
        List of nsample-dimensional data points.  Each row
        corresponds to a single data point.
        
    pdf_X_validate : array-like of shape (nsample, )
        Pdf of X_validate.
    """
    
    dim=X.shape[1]
    
    if kwargs["method"]=="bounded":
        lower=np.array([np.quantile(X[:,i],kwargs["ruleout"]) for i in range(dim)])
        upper=np.array([np.quantile(X[:,i],1-kwargs["ruleout"]) for i in range(dim)])
            
        np.random.seed(kwargs["seed"])
        return np.random.rand(int(nsample),dim)*(upper-lower)+lower,np.ones(int(nsample))/np.prod(upper-lower)
    if kwargs["method"]=="heavy_tail":
        density=TDistribution(loc=np.zeros(dim),scale=np.ones(dim),df=2/3)
        np.random.seed(kwargs["seed"])
        return density.generate(int(nsample))
    if kwargs["method"]=="normal":
        density=MultivariateNormalDistribution(mean=X.mean(axis=0),cov=np.diag(np.diag(np.cov(X.T))))
        np.random.seed(kwargs["seed"])
        return density.generate(int(nsample))
    
    if kwargs["method"]=="mixed":
        density1 = MultivariateNormalDistribution(mean=X.mean(axis=0),cov=np.diag(np.diag(np.cov(X.T)))) 
        density2 = TDistribution(loc=np.zeros(dim),scale=np.ones(dim),df=2/3)
        density_seq = [density1, density2]
        prob_seq = [0.7,0.3]
        densitymix = MixedDistribution(density_seq, prob_seq)
        return densitymix.generate(int(nsample))
    
    
  
def aknn(X,tree,n,dim,vol_unitball,kmax,C,beta):
    """Balanced k-NN density estimation. 

    Parameters
    ----------
    X : array-like of shape (n_test, dim_)
        List of n_test-dimensional data points.  Each row
        corresponds to a single data point.
        
    tree_ : "KDTree" instance
        The tree algorithm for fast generalized N-point problems.
        
    kmax : int
        Number of maximum neighbors to consider in estimaton. 
        
    n : int
        Number of traning samples.
        
    dim : int
        Number of features.
        
    vol_unitball : float
        Volume of dim_ dimensional unit ball.
        
    kmax : int
        Number of maximum neighbors to consider in estimaton. 
        
    C : float 
        Scaling paramerter in BKNN.
        
    C2 : float 
        Threshold paramerter in BKNN.
     

    Returns
    -------
    log_density: array-like of shape (n_test, ).
        Estimated log-density of test samples.
        
    """

    if len(X.shape)==1:
        X=X.reshape(1,-1).copy()
    
    
    log_density=[]
    
    
    for x in X:
        
        distance_vec,_=tree.query(x.reshape(1,-1),kmax)
        
        if distance_vec[0,0]==0:
            distance_vec=distance_vec[1:]
        k_temp=1
        while distance_vec[0,k_temp-1]**beta*k_temp<C and k_temp<kmax:
            k_temp+=1

        log_density.append(np.log(k_temp/n/vol_unitball/(distance_vec[0,k_temp]**dim)+1e-30))
   
        
    return np.array(log_density)
    
    