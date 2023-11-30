def READ_JRA55(season,tvar,tlev):
    import os
    import xarray as xr
    import warnings
    warnings.filterwarnings('ignore')

    # tvar="sic"
    # tlev="0"
    # tlev="all"
    # tvar ="tmp2m" #hgt tmp2m, sic, tmp, ugrd, vgrd
    # season = "DJ"
    file_path=os.path.abspath(__file__)
    wdir=f"{os.path.dirname(file_path)}/../"

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
        # syr = 1993
        # eyr = 2016
        syr = 1958
        eyr = 2013
        tmon = [12, 1, 2]

    nmon = len(tmon)

    nyrs = eyr-syr+1

    if tvar == "sic":
        inDir = wdir+"/DATA/HadISST_ice/HadISST_ice."
    else:
        inDir = wdir+"/DATA/JRA55/"+tvar+"/"+tvar+".day."
    data_arrays=[]
    for iy in range(nyrs):
        for im in range(nmon):
            if tmon[im] < tmon[0]:
                year = syr + iy + 1
            else:
                year = syr + iy

            inFname = f"{inDir}{year}{tmon[im]:02d}.nc"
            print(inFname)  # 파일 이름 출력

            inf1 = xr.open_dataset(inFname)

            if tvar in ["hgt", "ugrd", "vgrd", "tmp"]:
                if tlev == "all":
                    jvar = inf1[tvar][:, inf1['lev']>=10, :, :]
                else:
                    jvar = inf1[tvar][:, inf1['lev']==tlev, :, :]
            elif tvar == "tmp2m":
                jvar = inf1[tvar][:, :, :]
            elif tvar == "sic":
                jvar = inf1[tvar]
            leap_day_mask = (jvar['time'].dt.month == 2) & (jvar['time'].dt.day == 29)
            jvar = jvar.sel(time=~leap_day_mask)
            data_arrays.append(jvar)
    # xarray.concat으로 데이터 연결
    ovar_tmp = xr.concat(data_arrays, dim="time")
    if 'lat' and 'lon' in ovar_tmp.dims:
        ovar_tmp=ovar_tmp.rename({'lat':'latitude','lon':'longitude'})    
    dec_jan_mask = ovar_tmp['time.month'].isin(tmon)
    dec_jan_ds = ovar_tmp.where(dec_jan_mask, drop=True)
    dec_jan_ds['year'] = dec_jan_ds['time.year']
    dec_jan_ds['year'] = xr.where(dec_jan_ds['time.month'] == 12, dec_jan_ds['year'] + 1, dec_jan_ds['year'])
    annual_means = dec_jan_ds.groupby('year').mean('time')

    annual_means=annual_means.squeeze()
    print(annual_means)
    return annual_means


def READ_GloSea(season,tvar,tlev,imodel):
    import os
    import numpy as np
    import xarray as xr
    import warnings
    warnings.filterwarnings('ignore')

    # season = "DJ"
    # tvar = "temp"
    # tlev = 500
    # imodel="GloSea5"
    file_path=os.path.abspath(__file__)
    wdir=f"{os.path.dirname(file_path)}/../"

    
    if season == "DJ":
        syr, eyr = 1993, 2015
        tmon = [12, 1]
    elif season == "FM":
        syr, eyr = 1994, 2016
        tmon = [2, 3]
    elif season == "ON":
        syr, eyr = 1993, 2015
        tmon = [10, 11]

    opath = wdir+"/DATA/"+imodel+"/POST/"

    tfiles = [f"{opath}/{tvar}/{tvar}.{syr+nn}{tmon[0]:02d}01_ensmean.nc" for nn in range(eyr - syr + 1)]
    print(*tfiles,sep='\n')
    f_combined=[]
    for fname in tfiles:
        f=xr.open_dataset(fname)
        try:
            f=f.rename({'t':'time'})
        except:
            pass
        f=f.sel(time=f['time'].dt.month.isin(tmon))
        leap_day_mask = (f['time'].dt.month == 2) & (f['time'].dt.day == 29)
        f = f.sel(time=~leap_day_mask)
        f_combined.append(f)
        
    f_combined = xr.concat(f_combined, dim='time')

    if tvar is 't15m':
        data=f_combined['t15m']
    else:
        var=list(f_combined.data_vars)[0]
        data=f_combined[var]

    if 'lat' and 'lon' in data.dims:
        data=data.rename({'lat':'latitude','lon':'longitude'})    

    if 'level' in data.dims:
        data=data.rename({'level':'lev'})
    if 'lev' in data.dims:
        if tlev is not "all":
            data = data[:, data['lev']==tlev, :, :]

    data['year'] = data['time.year']
    data['year'] = xr.where(data['time.month'] == 12, data['year'] + 1, data['year'])
    annual_means = data.groupby('year').mean('time')
    annual_means=annual_means.squeeze()
    print(annual_means)
    return annual_means
