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