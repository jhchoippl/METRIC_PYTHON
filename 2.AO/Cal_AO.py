def cal_ano(var):
    var_clim = var.mean('year')
    var_ano = var - var_clim
    return var_ano.astype('float32')


def regres(hgt_tmp, eof_regres_jra, season, lat_jra, lon_jra):
    import numpy as np
    import statsmodels.api as sm
    import xarray as xr
    
    if season == "DJ":
        syr, eyr = 1993, 2015
        tmon = [12, 1]
    elif season == "FM":
        syr, eyr = 1994, 2016
        tmon = [2, 3]
    elif season == "ON":
        syr, eyr = 1993, 2015
        tmon = [10, 11]
    elif season == "DJF":
        syr, eyr = 1958, 2013
        tmon = [12, 1, 2]

    nyrs = eyr - syr + 1
    hgt = hgt_tmp    
    ano = cal_ano(hgt)

    dx = np.zeros((len(lat_jra) * len(lon_jra), nyrs))
    for i in range(nyrs):
        dx[:, i] = ano.isel(year=i).values.flatten()

    xmode = eof_regres_jra.isel(mode=0).values.flatten()

    ex = np.zeros(nyrs)
    for i in range(nyrs):
        ex[i] = np.sum(xmode * dx[:, i])

    ex_mean = np.mean(ex)
    ex_std = np.std(ex)
    nex = (ex - ex_mean) / ex_std

    eof_regres = xr.full_like(ano.isel(year=0), fill_value=np.nan)
    for lat_idx in range(len(ano.latitude)):
        for lon_idx in range(len(ano.longitude)):
            y = ano.isel(latitude=lat_idx, longitude=lon_idx).data
            model = sm.OLS(y, nex).fit()
            eof_regres[lat_idx, lon_idx] = model.params[0]

    return nex, eof_regres

def calculate_eof_ts_jra(season, eof_regres_jra, lat_jra, lon_jra):
    import sys
    import numpy as np
    import xarray as xr
    import statsmodels.api as sm
    sys.path.append("../src")
    import READ_FILE 
    
    tvar = "hgt"
    tlev = 1000
    ovar = READ_FILE.READ_JRA55(season, tvar, tlev)
    nyrs = len(ovar.year)
    hgt = ovar[:, 20 <= ovar['latitude'], :]
    ano = cal_ano(hgt)

    dx = np.zeros((len(lat_jra) * len(lon_jra), nyrs))
    for i in range(nyrs):
        dx[:, i] = ano.isel(year=i).values.flatten()

    xmode = eof_regres_jra.isel(mode=0).values.flatten()

    ex = np.zeros(nyrs)
    for i in range(nyrs):
        ex[i] = np.sum(xmode * dx[:, i])

    ex_mean = np.mean(ex)
    ex_std = np.std(ex)
    nex = (ex - ex_mean) / ex_std

    return nex