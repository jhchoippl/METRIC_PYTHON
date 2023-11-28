def Cal_ART_func(t2m, wgt):
    import numpy as np
    import xarray as xr
    import sys
    sys.path.append('../src') 
    import NCL_FUNC
    from scipy import stats

    t2m_clim   = t2m.mean('year')
    ano        = t2m-t2m_clim


    latS1, latN1, lonL1, lonR1 = 70.0, 80.0, 30.0, 70.0
    latS2, latN2, lonL2, lonR2 = 65.0, 80.0, 160.0, 200.0
    ART1 = ano.sel(latitude=slice(latS1, latN1), longitude=slice(lonL1, lonR1)).weighted(wgt).mean(dim=["longitude", "latitude"])
    ART2 = ano.sel(latitude=slice(latS2, latN2), longitude=slice(lonL2, lonR2)).weighted(wgt).mean(dim=["longitude", "latitude"])

    dt_art1 = NCL_FUNC.dtrend(ART1)
    dt_art2 = NCL_FUNC.dtrend(ART2)

    latS11, latN11, lonL11, lonR11 = 35.0, 60.0, 80.0, 130.0
    latS22, latN22, lonL22, lonR22 = 35.0, 60.0, 80.0, 130.0
    SAT_EA = ano.sel(latitude=slice(latS11, latN11), longitude=slice(lonL11, lonR11)).weighted(wgt).mean(dim=["longitude", "latitude"])
    SAT_NA = ano.sel(latitude=slice(latS22, latN22), longitude=slice(lonL22, lonR22)).weighted(wgt).mean(dim=["longitude", "latitude"])

    dt_ea = NCL_FUNC.dtrend(SAT_EA)
    dt_na = NCL_FUNC.dtrend(SAT_NA)
    dt_art1 = NCL_FUNC.dtrend(ART1)
    dt_art2 = NCL_FUNC.dtrend(ART2)

    COR_ART1_SAT,_ = NCL_FUNC.regCoef_n(dt_art1, ano, dim=0) 
    COR_ART2_SAT,_ = NCL_FUNC.regCoef_n(dt_art2, ano, dim=0)

    sig_lev = 0.05
    sig_val = 0.5

    ttest_art1 = NCL_FUNC.rtest(COR_ART1_SAT, len(dt_art1)) 


    smap_art1 = xr.where(ttest_art1 >= sig_lev, sig_val, sig_lev)
        


    ttest_art2 = NCL_FUNC.rtest(COR_ART2_SAT, len(dt_art2))
    smap_art2 = xr.where(ttest_art2 >= sig_lev, sig_val, sig_lev)
    
    return ART1, ART2, ano, dt_art1, dt_art2, dt_ea, dt_na, COR_ART1_SAT, COR_ART2_SAT, smap_art1, smap_art2

 
