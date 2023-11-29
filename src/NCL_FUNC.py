def dtrend(data):
    import numpy as np
    coefficients = np.polyfit(data['year'], data, 1.25)
    trend = np.polyval(coefficients, data['year'])
    detrended_data = data - trend
    return detrended_data

def latRegWgt(lat):
    import numpy as np
    dlat = np.abs(np.diff(lat))
    dlat = np.concatenate(([dlat[0]], dlat))
    weights = np.sin(np.deg2rad(lat + dlat/2)) - np.sin(np.deg2rad(lat - dlat/2))
    weights /= weights.sum()/2
    weights[0]=weights[-1]=5.9495e-05
    return weights

def escorc(x,y):
    from scipy.stats import pearsonr
    corr, p_value = pearsonr(x,y)
    return corr, p_value

def escorc_n(x,y):
    import xarray as xr
    return xr.apply_ufunc(escorc,
                          x,
                          y,
                          input_core_dims=[['year'],['year']],
                          output_core_dims=[[], []],
                          vectorize=True,
                          dask='parallelized',
                          output_dtypes=[float,float])
    
def dim_rmsd_n(x, y, dim='year'):
    import numpy as np
    rmse = ((x - y) ** 2).mean(dim=dim)
    return np.sqrt(rmse)

def dim_rmsd_n_Wrap(da1, da2, dim='year'):
    rmsd_values = dim_rmsd_n(da1, da2, dim=dim)
    return rmsd_values

### Add the function by jechoi =========================================================================
def regCoef_n(x, y, dim=0):
    import numpy as np
    # x: independent variable (predictor)
    # y: dependent variable (response)
    # dim: dimension along which to calculate regression coefficients
    
    # Calculate regression coefficients along the specified dimension
    n = x.shape[dim]
    sum_x = np.sum(x, axis=dim)
    sum_y = np.sum(y, axis=dim)
    sum_xy = np.sum(x * y, axis=dim)
    sum_x_squared = np.sum(x**2, axis=dim)

    # Calculate coefficients (slope and intercept)
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x**2)
    intercept = (sum_y - slope * sum_x) / n

    return slope, intercept


def rtest(correlation_coefficient, n):
    import xarray as xr
    import numpy as np
    from scipy import stats

    t_statistic = correlation_coefficient * np.sqrt((n - 2) / (1 - correlation_coefficient ** 2))
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df=n - 2))
    
    return xr.DataArray(p_value, coords=correlation_coefficient.coords, dims=correlation_coefficient.dims)



def pattern_cor(x, y, weights):
    import numpy as np
    """
    Calculate the pattern correlation coefficient between two 2D arrays.

    Parameters:
    x (numpy.ndarray): First 2D array
    y (numpy.ndarray): Second 2D array
    weights (numpy.ndarray): 1D array of weights, typically latitude weights

    Returns:
    float: Pattern correlation coefficient
    """
    x=x.values
    y=y.values
    weights=weights.values
    # Ensure weights are normalized
    
    weights = weights / np.sum(weights)

    # Flatten the arrays
    x_flat = x.flatten()
    y_flat = y.flatten()
    weights_flat = np.repeat(weights, x.shape[1]).flatten()

    # Remove the mean
    x_mean_removed = x_flat - np.average(x_flat, weights=weights_flat)
    y_mean_removed = y_flat - np.average(y_flat, weights=weights_flat)

    # Calculate the weighted covariance and variances
    covariance = np.sum(weights_flat * x_mean_removed * y_mean_removed)
    variance_x = np.sum(weights_flat * x_mean_removed ** 2)
    variance_y = np.sum(weights_flat * y_mean_removed ** 2)

    # Calculate the pattern correlation coefficient
    pattern_correlation = covariance / np.sqrt(variance_x * variance_y)

    return pattern_correlation