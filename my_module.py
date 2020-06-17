from numpy import sqrt

def ERP(data):
    '''
    Compute the evoked response potential (ERP) with confidence bounds 

    Parameters
    ----------
    data : ndarray
        An array arranged as trials-by-samples

    Returns
    -------
    erp : ndarray
        the ERP of the data
    conf_lo : ndarray
        the lower confidence bound 
    conf_hi : ndarray
        the upper confidence bound
    '''
    
    N = len(data)  # Define the number of trials
    erp = data.mean(0)  # Compute the mean,
    sd = data.std(0)  # ... standard deviation,
    sdmn = sd / sqrt(N)  # ... and standard deviation of the mean
    conf_lo = erp - 2 * sdmn  # Computer the lower confidence bound,
    conf_hi = erp + 2 * sdmn  # ... and thee upper confidence bound

    return erp, conf_lo, conf_hi