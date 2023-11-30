import sys, os
import gc
import numpy as np
import xarray as xr
import geocat.viz as gv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

# 계절, 모델 setting
season="DJ"
try:
    season=sys.argv[1]
except IndexError:
    pass
print(season)
wdir='./'



model1 = "JRA55"
model2 = "GloSea5"
nens2  = 3
model3 = "GloSea6"
nens3  = 7

# 경도 범위 설정
lonL1, lonR1 = 20, 70
lonL2, lonR2 = 80, 130
lonL3, lonR3 = 130, 190

# 시즌에 따른 설정
if season == "DJ":
    syr, eyr = 1993, 2015
    init = "1201"
    tmon = [12,1]
elif season == "FM":
    syr, eyr = 1994, 2016
    init = "0201"
    tmon = [2,3]
elif season == "ON":
    syr, eyr = 1993, 2015
    init = "1001"
    tmon = [10,11]



yy = np.arange(syr, eyr + 1, 1)
nyy = len(yy)

yyindex_jra_list  = []
yyindex_Glo5_list = []
yyindex_Glo6_list = []

# JRA55 data setting
for year in yy:
    file_path     = f"{wdir}/{model1}/index/{model1}.{year}{init}_index_TM.nc"
    print(file_path)
    ds            = xr.open_dataset(file_path)
    index_data    = ds.sel(time=ds['time'].dt.month.isin(tmon))
    leap_day_mask = (index_data['time'].dt.month == 2) & (index_data['time'].dt.day == 29)
    index_data    = index_data.sel(time=~leap_day_mask)
    yyindex_sum   = index_data.sum(dim="time") 
    yyindex_prct  = (yyindex_sum / len(index_data['time'])) * 100.0 
    yyindex_jra_list.append(yyindex_prct)

yyindex_jra = xr.concat(yyindex_jra_list, dim="year")
del (file_path,ds,index_data,leap_day_mask,yyindex_sum,yyindex_prct,yyindex_jra_list)
gc.collect()

# Glosea5 data setting
for year in yy:
    yyindex_tmp_Glo5_list = []

    for enums in range(1, nens2 + 1):
        file_num      = f"{enums:02d}"
        file_path     = f"{wdir}/{model2}/index/{model2}.{year}{init}_index_TM_{file_num}.nc"
        print(file_path)
        ds            = xr.open_dataset(file_path)
        index_data    = ds.sel(time=ds['time'].dt.month.isin(tmon))
        leap_day_mask = (index_data['time'].dt.month == 2) & (index_data['time'].dt.day == 29)
        index_data    = index_data.sel(time=~leap_day_mask)
        yyindex_tmp_Glo5_list.append(index_data)

    yyindex_tmp_combined = xr.concat(yyindex_tmp_Glo5_list, dim="ensemble")
    yyindex_sum          = yyindex_tmp_combined.sum(dim=["time", "ensemble"])
    yyindex_prct         = (yyindex_sum / (len(index_data['time']) * nens2)) * 100.0
    yyindex_Glo5_list.append(yyindex_prct)

yyindex_gc2 = xr.concat(yyindex_Glo5_list, dim="year")
del (year,enums)
del (file_num,file_path,ds,index_data,leap_day_mask,yyindex_tmp_Glo5_list)
del (yyindex_tmp_combined,yyindex_sum,yyindex_prct,yyindex_Glo5_list)
gc.collect()

# Glosea6 data setting
for year in yy:
    yyindex_tmp_Glo6_list = []

    for enums in range(1, nens3 + 1):
        file_num      = f"{enums:02d}"
        file_path     = f"{wdir}/{model3}/index/{model3}.{year}{init}_index_TM_{file_num}.nc"
        print(file_path)
        ds            = xr.open_dataset(file_path)
        index_data    = ds.sel(time=ds['time'].dt.month.isin(tmon))
        leap_day_mask = (index_data['time'].dt.month == 2) & (index_data['time'].dt.day == 29)
        index_data    = index_data.sel(time=~leap_day_mask)
        yyindex_tmp_Glo6_list.append(index_data)

    yyindex_tmp_combined = xr.concat(yyindex_tmp_Glo6_list, dim="ensemble")
    yyindex_sum          = yyindex_tmp_combined.sum(dim=["time", "ensemble"])
    yyindex_prct         = (yyindex_sum / (len(index_data['time']) * nens3)) * 100.0
    yyindex_Glo6_list.append(yyindex_prct)

yyindex_gc32 = xr.concat(yyindex_Glo6_list, dim="year")
del (year,enums)
del (file_num,file_path,ds,index_data,leap_day_mask,yyindex_tmp_Glo6_list)
del (yyindex_tmp_combined,yyindex_sum,yyindex_prct,yyindex_Glo6_list)
gc.collect()

# 경도 평균 설정
fyyindex_ub_jra  = yyindex_jra.sel(lon=slice(lonL1,lonR1)).mean(dim='lon')
fyyindex_ub_gc2  = yyindex_gc2.sel(lon=slice(lonL1,lonR1)).mean(dim='lon')
fyyindex_ub_gc32 = yyindex_gc32.sel(lon=slice(lonL1,lonR1)).mean(dim='lon')

fyyindex_esb_jra  = yyindex_jra.sel(lon=slice(lonL2,lonR2)).mean(dim='lon')
fyyindex_esb_gc2  = yyindex_gc2.sel(lon=slice(lonL2,lonR2)).mean(dim='lon')
fyyindex_esb_gc32 = yyindex_gc32.sel(lon=slice(lonL2,lonR2)).mean(dim='lon')

fyyindex_kb_jra  = yyindex_jra.sel(lon=slice(lonL3,lonR3)).mean(dim='lon')
fyyindex_kb_gc2  = yyindex_gc2.sel(lon=slice(lonL3,lonR3)).mean(dim='lon')
fyyindex_kb_gc32 = yyindex_gc32.sel(lon=slice(lonL3,lonR3)).mean(dim='lon')

# Correlation 설정
cor_ub_gc2   = xr.corr(fyyindex_ub_jra['index'],fyyindex_ub_gc2['index'],dim="year")
cor_ub_gc32  = xr.corr(fyyindex_ub_jra['index'],fyyindex_ub_gc32['index'],dim="year")
cor_esb_gc2  = xr.corr(fyyindex_esb_jra['index'],fyyindex_esb_gc2['index'],dim="year")
cor_esb_gc32 = xr.corr(fyyindex_esb_jra['index'],fyyindex_esb_gc32['index'],dim="year")
cor_kb_gc2   = xr.corr(fyyindex_kb_jra['index'],fyyindex_kb_gc2['index'],dim="year")
cor_kb_gc32  = xr.corr(fyyindex_kb_jra['index'],fyyindex_kb_gc32['index'],dim="year")

# NRMSE 설정

nrmse_ub_gc2   = mean_squared_error(fyyindex_ub_jra['index'],fyyindex_ub_gc2['index'],squared=False)/(fyyindex_ub_jra['index'].max()-fyyindex_ub_jra['index'].min())
nrmse_ub_gc32  = mean_squared_error(fyyindex_ub_jra['index'],fyyindex_ub_gc32['index'],squared=False)/(fyyindex_ub_jra['index'].max()-fyyindex_ub_jra['index'].min())
nrmse_esb_gc2  = mean_squared_error(fyyindex_esb_jra['index'],fyyindex_esb_gc2['index'],squared=False)/(fyyindex_esb_jra['index'].max()-fyyindex_esb_jra['index'].min())
nrmse_esb_gc32 = mean_squared_error(fyyindex_esb_jra['index'],fyyindex_esb_gc32['index'],squared=False)/(fyyindex_esb_jra['index'].max()-fyyindex_esb_jra['index'].min())
nrmse_kb_gc2   = mean_squared_error(fyyindex_kb_jra['index'],fyyindex_kb_gc2['index'],squared=False)/(fyyindex_kb_jra['index'].max()-fyyindex_kb_jra['index'].min())
nrmse_kb_gc32  = mean_squared_error(fyyindex_kb_jra['index'],fyyindex_kb_gc32['index'],squared=False)/(fyyindex_kb_jra['index'].max()-fyyindex_kb_jra['index'].min())

jra_datasets  = [fyyindex_ub_jra, fyyindex_esb_jra, fyyindex_kb_jra]
gc2_datasets  = [fyyindex_ub_gc2, fyyindex_esb_gc2, fyyindex_kb_gc2]
gc32_datasets = [fyyindex_ub_gc32, fyyindex_esb_gc32, fyyindex_kb_gc32]

gc2_CORNRMSE1 = f"GloSea5(R={cor_ub_gc2:.2f}/NRMSE={nrmse_ub_gc2:.2f})"
gc2_CORNRMSE2 = f"GloSea5(R={cor_esb_gc2:.2f}/NRMSE={nrmse_esb_gc2:.2f})"
gc2_CORNRMSE3 = f"GloSea5(R={cor_kb_gc2:.2f}/NRMSE={nrmse_kb_gc2:.2f})"
gc2_CORNRMSE  = [gc2_CORNRMSE1,gc2_CORNRMSE2,gc2_CORNRMSE3]

gc32_CORNRMSE1 = f"GloSea6(R={cor_ub_gc32:.2f}/NRMSE={nrmse_ub_gc32:.2f})"
gc32_CORNRMSE2 = f"GloSea6(R={cor_esb_gc32:.2f}/NRMSE={nrmse_esb_gc32:.2f})"
gc32_CORNRMSE3 = f"GloSea6(R={cor_kb_gc32:.2f}/NRMSE={nrmse_kb_gc32:.2f})"
gc32_CORNRMSE  = [gc32_CORNRMSE1,gc32_CORNRMSE2,gc32_CORNRMSE3]

del (fyyindex_ub_jra,fyyindex_ub_gc2,fyyindex_ub_gc32)
del (fyyindex_esb_jra,fyyindex_esb_gc2,fyyindex_esb_gc32)
del (fyyindex_kb_jra,fyyindex_kb_gc2,fyyindex_kb_gc32)
del (cor_ub_gc2,cor_ub_gc32,cor_esb_gc2,cor_esb_gc32,cor_kb_gc2,cor_kb_gc32)
del (nrmse_ub_gc2,nrmse_ub_gc32,nrmse_esb_gc2,nrmse_esb_gc32,nrmse_kb_gc2,nrmse_kb_gc32)
gc.collect()

syr, eyr = 1993, 2015

fig = plt.figure(figsize=(10, 10),dpi=300)

grid = gridspec.GridSpec(nrows=3, ncols=1, figure=fig,hspace=0.4, wspace= 0.5)
ax1 = fig.add_subplot(grid[0,:])
ax2 = fig.add_subplot(grid[1,:])
ax3 = fig.add_subplot(grid[2,:])

axs=[ax1,ax2,ax3]
time = np.arange(syr, eyr + 1)
blk_n=['Ural','East Siberian','Kamchatka']
for i in [0,1,2]:
    axs[i].plot(time, jra_datasets[i]['index'].values, label='HadISST', color='black', marker='o')
    axs[i].plot(time, gc2_datasets[i]['index'].values, color='red', marker='o')
    axs[i].plot(time, gc32_datasets[i]['index'].values, color='blue', marker='o')

    axs[i].grid(True, which='major', linestyle= (0,(5,3,5,10)), linewidth=0.6)
    axs[i].set_xlabel('Year', fontsize=13)  # Corrected method
    axs[i].set_ylabel('[%]', fontsize=13)  # Corrected metho
    gv.set_titles_and_labels(axs[i],
                            lefttitle=f'{blk_n[i]} Blocking',
                            righttitle=f'[{season}, {syr}/{syr+1}-{eyr}/{eyr+1}]',
                            lefttitlefontsize=15,
                            righttitlefontsize=10)
    axs[i].legend(labels=[
        'JRA55',
        gc2_CORNRMSE[i],
        gc32_CORNRMSE[i]
    ], fontsize=8, frameon=False)

    axs[i].set_xlim([min(time),max(time)])
    if i ==1:
        if season=='DJ':
            axs[i].set_ylim(0,20)
        else:
            axs[i].set_ylim(0,10)
    else:
        axs[i].set_ylim(0,40)
    axs[i].set_xticks(np.arange(syr, eyr,4))

ofile=f"./Figure/3-1_Blocking_{season}.png"
plt.savefig(ofile, bbox_inches='tight')
plt.close()