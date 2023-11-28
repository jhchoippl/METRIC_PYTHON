#!/home/jhchoi/anaconda3/envs/py311/bin/python

import sys, os
import xarray as xr
import numpy as np
from eofs.xarray import Eof
import statsmodels.api as sm

import Cal_AO
sys.path.append("../src")
import READ_FILE 
import NCL_FUNC 

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmaps
import geocat.viz as gv

import warnings
warnings.filterwarnings('ignore')


def add_cyclic(data):
    import geocat.viz as gv
    return gv.xr_add_cyclic_longitudes(data,'longitude')

season="FM"

try:
    season=sys.argv[1]
except IndexError:
    pass
print(season)

wdir  = "/data03/Glosea5/jhsim/NCL2PYTHON/METRIC_NCL/"
inDir = f'{wdir}/DATA/JRA55/hgt_mon/'

syr  = 1979
eyr  = 2000
nyrs = eyr - syr + 1
neof = 1
latS, latN = 20, 90 

hgt=[]

for iy in range(nyrs):
    year    = syr + iy
    inFname = f"{inDir}hgt.mon.{year}.nc"
    inf     = xr.open_dataset(inFname)
    ihgt    = inf['hgt']
    jhgt    = ihgt[:,ihgt['lev']==1000,:,:]
    hgt.append(jhgt)
hgt=xr.concat(hgt, dim="time")

hgt = hgt.squeeze('lev')
hgt = hgt[:,latS<=hgt['lat'],:]

# make climatology
hgt2 = hgt.groupby('time.month').mean('time')

# make anomaly
hgt_ano = hgt.groupby('time.month')-hgt2
hgt_ano = hgt_ano.drop('lev')

lat = hgt_ano.lat
lon = hgt_ano.lon

rad          = np.pi / 180.  # degree to rad
clat         = hgt_ano.lat
clat_weights = np.sqrt(np.cos(rad * clat))

# latitude weighting 
xw = hgt_ano * clat_weights

xw = xw.drop('month')

#======= EOF ============================ 
solver = Eof(xw)
eof    = -1*solver.eofs(neofs=1, eofscaling=0) # EOFs are multiplied by the square-root of their eigenvalues
eof_ts = -1*solver.pcs(npcs=1, pcscaling=1) # PCs are scaled to unit variance

varfrac = solver.varianceFraction()*100

# eof와 동일한 구조를 가진 빈 DataArray 생성
eof_regres = xr.full_like(eof, fill_value=np.nan)

# 각 EOF 모드에 대한 회귀 계수 계산
for ne in range(neof):
    # eof_ts의 현재 모드 선택
    current_eof_ts = eof_ts.sel(mode=ne).data
    
    # xAnom과 현재 eof 모드와의 회귀 분석 수행
    for lat_idx in range(len(hgt_ano.lat)):
        for lon_idx in range(len(hgt_ano.lon)):
            y = hgt_ano.isel(lat=lat_idx, lon=lon_idx).data
            model = sm.OLS(y, current_eof_ts).fit()
            eof_regres[ne, lat_idx, lon_idx] = model.params[0]
            

eof_regres.attrs['pcvar'] = float(varfrac[:neof].item())

eof_regres = eof_regres.rename({'lat': 'latitude', 'lon': 'longitude'})

# 결과 출력
eof_regres_jra = eof_regres


lat_jra = lat
lon_jra = lon

del nyrs, lat, lon, eof_ts, eof_regres, hgt, hgt2, solver, hgt_ano, clat, xw, eof

if season == "DJ":
    syr, eyr = 1993, 2015
    tmon = [12, 1]
elif season == "FM":
    syr, eyr = 1994, 2016
    tmon = [2, 3]
elif season == "ON":
    syr, eyr = 1993, 2015
    tmon = [10, 11]

eof_ts_jra = Cal_AO.calculate_eof_ts_jra(season, eof_regres_jra, lat_jra, lon_jra)

model0 = "GloSea5"
imodel = model0
tlev   = 1000
tvar   = "hgt"
mvar = READ_FILE.READ_GloSea(season,tvar,tlev,imodel)

hgt_tmp=mvar[:,20<=mvar['latitude'],:]
eof_ts_gc2,eof_regres_gc2 = Cal_AO.regres(hgt_tmp, eof_regres_jra, season, lat_jra, lon_jra)

model1 = "GloSea6"
imodel = model1
tlev   = 1000
tvar   = "hgt"
mvar = READ_FILE.READ_GloSea(season,tvar,tlev,imodel)

hgt_tmp = mvar[:,20<=mvar['latitude'],:]
eof_ts_gc32,eof_regres_gc32 = Cal_AO.regres(hgt_tmp, eof_regres_jra, season, lat_jra, lon_jra)

mvar_avg_jra  = eof_ts_jra
mclim_avg_jra = mvar_avg_jra.mean()
mano_avg_jra  = mvar_avg_jra - mclim_avg_jra

mvar_avg_gc2  = eof_ts_gc2
mclim_avg_gc2 = mvar_avg_gc2.mean()
mano_avg_gc2  = mvar_avg_gc2 - mclim_avg_gc2

mvar_avg_gc32  = eof_ts_gc32
mclim_avg_gc32 = mvar_avg_gc32.mean()
mano_avg_gc32  = mvar_avg_gc32 - mclim_avg_gc32


eof_ts_jra  = xr.DataArray(eof_ts_jra, coords={'year': np.arange(syr, eyr + 1)}, dims='year')
eof_ts_gc2  = xr.DataArray(eof_ts_gc2, coords={'year': np.arange(syr, eyr + 1)}, dims='year')
eof_ts_gc32 = xr.DataArray(eof_ts_gc32, coords={'year': np.arange(syr, eyr + 1)}, dims='year')
eof_ts = [eof_ts_jra,eof_ts_gc2,eof_ts_gc32]

eof_regres_jra=eof_regres_jra.drop_vars('mode').squeeze('mode')
eof_regres_jra=add_cyclic(eof_regres_jra)

# dtrend 함수를 사용하여 추세 제거
deof_ts_jra  = NCL_FUNC.dtrend(eof_ts_jra)
deof_ts_gc32 = NCL_FUNC.dtrend(eof_ts_gc32)
deof_ts_gc2  = NCL_FUNC.dtrend(eof_ts_gc2)

# 상관 계수 계산
cor_ts_gc2 , _ = NCL_FUNC.escorc_n(eof_ts_jra, eof_ts_gc2)
cor_ts_gc32, _ = NCL_FUNC.escorc_n(eof_ts_jra, eof_ts_gc32)

# 추세를 제거한 데이터의 상관 계수 계산
dcor_ts_gc2 , _ = NCL_FUNC.escorc_n(deof_ts_jra, deof_ts_gc2)
dcor_ts_gc32, _ = NCL_FUNC.escorc_n(deof_ts_jra, deof_ts_gc32)

# RMSE 계산
rmse_ts_gc2  = NCL_FUNC.dim_rmsd_n(eof_ts_jra, eof_ts_gc2)
rmse_ts_gc32 = NCL_FUNC.dim_rmsd_n(eof_ts_jra, eof_ts_gc32)

# 정규화된 RMSE 계산
nrmse_ts_gc2  = rmse_ts_gc2 / (eof_ts_jra.max() - eof_ts_jra.min())
nrmse_ts_gc32 = rmse_ts_gc32 / (eof_ts_jra.max() - eof_ts_jra.min())

# 추세를 제거한 데이터의 RMSE 계산
drmse_ts_gc2  = NCL_FUNC.dim_rmsd_n(deof_ts_jra, deof_ts_gc2)
drmse_ts_gc32 = NCL_FUNC.dim_rmsd_n(deof_ts_jra, deof_ts_gc32)

# 추세를 제거한 데이터의 정규화된 RMSE 계산
dnrmse_ts_gc2  = drmse_ts_gc2 / (deof_ts_jra.max() - deof_ts_jra.min())
dnrmse_ts_gc32 = drmse_ts_gc32 / (deof_ts_jra.max() - deof_ts_jra.min())


fig = plt.figure(figsize=(10, 10),dpi=300)

grid = gridspec.GridSpec(nrows=3, ncols=2, figure=fig,hspace=0.4, wspace=-0.5)
proj = projection = ccrs.NorthPolarStereo()

ax1 = fig.add_subplot(grid[:-1,:], projection=proj)
ax2 = fig.add_subplot(grid[-1:,:])


ax1.coastlines(resolution='10m', color='black', linewidth=0.3)
ax1.add_feature(cfeature.LAKES, edgecolor='black', linewidth=0.3, facecolor='none')
gv.set_map_boundary(ax1, [-180, 180], [20, 90], south_pad=1)

gl = ax1.gridlines(ccrs.PlateCarree(),
                draw_labels=False,
                linestyle=(0,(1,2)),
                linewidth=0.5,
                color='black',
                zorder=2)

gl.ylocator = mticker.FixedLocator(np.arange(0, 90, 15))
gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 30))

ticks = np.arange(0, 210, 30)
etick = ['0'] + [f'{tick}E' for tick in ticks if (tick != 0) & (tick != 180)] + ['180']
wtick = [f'{tick}W' % tick for tick in ticks if (tick != 0) & (tick != 180)]
labels = [
    '0', '30E', '60E', '90E', '120E', '150E', '180', '150W', '120W',
    '90W', '60W', '30W'
]
xticks = np.arange(0, 360, 30)
yticks = np.full_like(xticks, 14)  # Latitude where the labels will be drawn

tick_size=10

for xtick, ytick, label in zip(xticks, yticks, labels):
    if label == '180':
        ax1.text(xtick,
                ytick,
                label,
                fontsize=tick_size,
                horizontalalignment='center',
                verticalalignment='top',
                transform=ccrs.Geodetic())
    elif label == '0':
        ax1.text(xtick,
                ytick,
                label,
                fontsize=tick_size,
                horizontalalignment='center',
                verticalalignment='bottom',
                transform=ccrs.Geodetic())
    else:
        ax1.text(xtick,
                ytick,
                label,
                fontsize=tick_size,
                horizontalalignment='center',
                verticalalignment='center',
                transform=ccrs.Geodetic())
gv.set_titles_and_labels(ax1,
                        maintitle=f'AO Pattern \n (JRA55, 1979-2000)', 
                        maintitlefontsize=20)


cmap=cmaps.BlueWhiteOrangeRed
v=list(range(-30,31,5))
C=ax1.contourf(
    eof_regres_jra['longitude'].data,
    eof_regres_jra['latitude'].data,
    eof_regres_jra.data,
    transform=ccrs.PlateCarree(),
    levels=v,
    cmap=cmap,
    extend='both'
    )


v= np.arange(-30, 31, 10)
cbar = plt.colorbar(C,
                    ax=ax1,
                    ticks=v,
                    extendfrac='auto',
                    aspect=15,
                    orientation='horizontal',
                    extend='both',
                    drawedges=True,
                    fraction=0.04
                    )
cbar.ax.tick_params(labelsize=10)

time = np.arange(syr, eyr + 1)
ax2.plot(time, eof_ts[0], label='HadISST', color='black', marker='o')
ax2.plot(time, eof_ts[1], color='red', marker='o')
ax2.plot(time, eof_ts[2], color='blue', marker='o')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.6)


ax2.grid(True, which='major', linestyle= (0,(5,3,5,10)), linewidth=0.6)
ax2.set_xlabel('Year', fontsize=13)  # Corrected method
ax2.set_ylabel('Standardized', fontsize=13)  # Corrected metho

gv.set_titles_and_labels(ax2,
                        lefttitle=f'AO Index (variance:{eof_regres_jra.pcvar:.1f}%)',
                        righttitle=f'[{season}, 1993/1994-2015/2016]',
                        lefttitlefontsize=12,
                        righttitlefontsize=12)


ax2.legend(labels=[
    'JRA55',
    f'GloSea5 (R={cor_ts_gc2.values:.2f}({dcor_ts_gc2.values:.2f})/NRMSE={nrmse_ts_gc2.values:.2f}({dnrmse_ts_gc2.values:.2f}))'
,
    f'GloSea5 (R={cor_ts_gc32.values:.2f}({dcor_ts_gc32.values:.2f})/NRMSE={nrmse_ts_gc32.values:.2f}({dnrmse_ts_gc32.values:.2f}))'

], fontsize=8, frameon=False)

# Adjusting the tick parameters
ax2.set_xlim([min(time),max(time)])
ax2.set_ylim(-3,4)

# ax2.set_yticks(np.arange(-4,5,2))
ax2.set_xticks(np.arange(syr, eyr,4))

ofile=f"./Figure/2-1_AO_{season}.png"
plt.savefig(ofile, bbox_inches='tight')
plt.close()