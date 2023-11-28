import sys, os
import numpy as np
import xarray as xr
import gc

from zmq import TYPE
sys.path.append("../src")
import NCL_FUNC
import READ_FILE
import Cal_ART

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmaps
import geocat.viz as gv

def add_cyclic(data):
    import geocat.viz as gv
    return gv.xr_add_cyclic_longitudes(data,'longitude')

def polar_proj_polyline(lat_min, lat_max, lon_min, lon_max):
    import numpy as np
    lower_left = np.column_stack((np.full(100, lon_min), np.linspace(lat_min, lat_max, 100)))
    lower_right = np.column_stack((np.full(100, lon_max), np.linspace(lat_min, lat_max, 100)))
    upper_left = np.column_stack((np.linspace(lon_min, lon_max, 100), np.full(100, lat_min)))
    upper_right = np.column_stack((np.linspace(lon_min, lon_max, 100), np.full(100, lat_max)))
    corners=[lower_left, lower_right, upper_left, upper_right]
    return corners    

season = "DJ"   # ON, DJ, FM
try:
    season=sys.argv[1]
except IndexError:
    pass
print(season)

# READ JRA55
tlev = 1000   # Not used
tvar = "tmp2m"
ot2m3 = READ_FILE.READ_JRA55(season,tvar,tlev)

nyrs=ot2m3['year'].size

# READ GloSea5
model0 = "GloSea5"
tlev = 1000
tvar = "t15m"
mt2m3_gc2 = READ_FILE.READ_GloSea(season,tvar,tlev,model0)

# READ GloSea6
model0 = "GloSea6"
tlev = 1000
tvar = "t15m"
mt2m3_gc32 = READ_FILE.READ_GloSea(season,tvar,tlev,model0)

# JRA55 
t2m= ot2m3
owgt=NCL_FUNC.latRegWgt(ot2m3['latitude'])
wgt  = owgt
clat = owgt
wgt.name='weights'


ART1, ART2, ano, dt_art1, dt_art2, dt_ea, dt_na, COR_ART1_SAT, COR_ART2_SAT, smap_art1, smap_art2= Cal_ART.Cal_ART_func(t2m, wgt)
       
dcor_art_jra=NCL_FUNC.escorc(dt_art1,dt_art2)
cor_art_jra=NCL_FUNC.escorc(ART1,ART2)

ART1_jra=ART1    #[year | 23]
ART2_jra=ART2
SAT_jra =ano

COR_ART1_SAT_jra = COR_ART1_SAT
COR_ART2_SAT_jra = COR_ART2_SAT
#print("COR_ART1_SAT ->",type(COR_ART1_SAT))

smap_art1_jra = smap_art1
smap_art2_jra = smap_art2

dt_art1_jra = dt_art1
dt_art2_jra = dt_art2

dt_ea_jra = dt_ea
dt_na_jra = dt_na

del ART1, ART2, ano, dt_art1, dt_art2, dt_ea, dt_na, COR_ART1_SAT, COR_ART2_SAT, smap_art1, smap_art2
gc.collect()

# Glosea5
t2m= mt2m3_gc2

ART1, ART2, ano, dt_art1, dt_art2, dt_ea, dt_na, COR_ART1_SAT, COR_ART2_SAT, smap_art1, smap_art2=Cal_ART.Cal_ART_func(t2m, wgt)
dcor_art_gc2=NCL_FUNC.escorc(dt_art1,dt_art2)
cor_art_gc2=NCL_FUNC.escorc(ART1,ART2)

ART1_gc2=ART1
ART2_gc2=ART2
SAT_gc2 =ano

COR_ART1_SAT_gc2 = COR_ART1_SAT
COR_ART2_SAT_gc2 = COR_ART2_SAT

smap_art1_gc2 = smap_art1
smap_art2_gc2 = smap_art2

dt_art1_gc2 = dt_art1
dt_art2_gc2 = dt_art2

dt_ea_gc2 = dt_ea
dt_na_gc2 = dt_na

del ART1, ART2, ano, dt_art1, dt_art2, dt_ea, dt_na, COR_ART1_SAT, COR_ART2_SAT, smap_art1, smap_art2
gc.collect()

#Glosea6
t2m = mt2m3_gc32

ART1, ART2, ano, dt_art1, dt_art2, dt_ea, dt_na, COR_ART1_SAT, COR_ART2_SAT, smap_art1, smap_art2=Cal_ART.Cal_ART_func(t2m, wgt)
dcor_art_gc32=NCL_FUNC.escorc(dt_art1,dt_art2)
cor_art_gc32=NCL_FUNC.escorc(ART1,ART2)

ART1_gc32=ART1
ART2_gc32=ART2
SAT_gc32 =ano

COR_ART1_SAT_gc32 = COR_ART1_SAT
COR_ART2_SAT_gc32 = COR_ART2_SAT

smap_art1_gc32 = smap_art1
smap_art2_gc32 = smap_art2

dt_art1_gc32 = dt_art1
dt_art2_gc32 = dt_art2

dt_ea_gc32 = dt_ea
dt_na_gc32 = dt_na

del ART1, ART2, ano, dt_art1, dt_art2, dt_ea, dt_na, COR_ART1_SAT, COR_ART2_SAT, smap_art1, smap_art2
gc.collect() 

# 1. MSSS

latS111    = 35.0
latN111    = 80.0
lonL111    = 30.0
lonR111    = 130.0

latS222    = 35.0
latN222    = 80.0
lonL222    = 160.0
lonR222    = 280.0

COR_ART1_SAT_jra_subset = COR_ART1_SAT_jra.sel(latitude=slice(latS111, latN111), longitude=slice(lonL111, lonR111))
COR_ART1_SAT_gc2_subset = COR_ART1_SAT_gc2.sel(latitude=slice(latS111, latN111), longitude=slice(lonL111, lonR111))
COR_ART1_SAT_gc32_subset = COR_ART1_SAT_gc32.sel(latitude=slice(latS111, latN111), longitude=slice(lonL111, lonR111))
clat_subset1 = clat.sel(latitude=slice(latS111, latN111))

pcc_art1_gc2  = NCL_FUNC.pattern_cor(COR_ART1_SAT_jra_subset, COR_ART1_SAT_gc2_subset, clat_subset1)
pcc_art1_gc32 =  NCL_FUNC.pattern_cor(COR_ART1_SAT_jra_subset, COR_ART1_SAT_gc32_subset, clat_subset1)

COR_ART2_SAT_jra_subset = COR_ART2_SAT_jra.sel(latitude=slice(latS222, latN222), longitude=slice(lonL222, lonR222))
COR_ART2_SAT_gc2_subset = COR_ART2_SAT_gc2.sel(latitude=slice(latS222, latN222), longitude=slice(lonL222, lonR222))
COR_ART2_SAT_gc32_subset = COR_ART2_SAT_gc32.sel(latitude=slice(latS222, latN222), longitude=slice(lonL222, lonR222))
clat_subset2 = clat.sel(latitude=slice(latS222, latN222))

pcc_art2_gc2  = NCL_FUNC.pattern_cor(COR_ART2_SAT_jra_subset, COR_ART2_SAT_gc2_subset, clat_subset2)
pcc_art2_gc32 =  NCL_FUNC.pattern_cor(COR_ART2_SAT_jra_subset, COR_ART2_SAT_gc32_subset, clat_subset2)

rmse_map1_gc2 = np.sqrt( ((COR_ART1_SAT_jra_subset-COR_ART1_SAT_gc2_subset)**2).weighted(clat_subset1).mean() )
rmse_map1_gc32 = np.sqrt( ((COR_ART1_SAT_jra_subset-COR_ART1_SAT_gc32_subset)**2).weighted(clat_subset1).mean() )

rmse_map2_gc2 = np.sqrt( ((COR_ART2_SAT_jra_subset-COR_ART2_SAT_gc2_subset)**2).weighted(clat_subset2).mean() )
rmse_map2_gc32 = np.sqrt( ((COR_ART2_SAT_jra_subset-COR_ART2_SAT_gc32_subset)**2).weighted(clat_subset2).mean() )

nrmse_map1_gc2 = rmse_map1_gc2/(COR_ART1_SAT_jra_subset.max()-COR_ART1_SAT_jra_subset.min())
nrmse_map1_gc32 = rmse_map1_gc32/(COR_ART1_SAT_jra_subset.max()-COR_ART1_SAT_jra_subset.min())

nrmse_map2_gc2 = rmse_map2_gc2/(COR_ART2_SAT_jra_subset.max()-COR_ART2_SAT_jra_subset.min())
nrmse_map2_gc32 = rmse_map2_gc32/(COR_ART2_SAT_jra_subset.max()-COR_ART2_SAT_jra_subset.min())


nyrs = len(ART1_jra)

mvar_avg_jra = ART1_jra 
mclim_avg_jra    = mvar_avg_jra.mean()  
mano_avg_jra     = mvar_avg_jra - mclim_avg_jra  

mvar_avg_gc2 = ART1_gc2
mclim_avg_gc2    = mvar_avg_gc2.mean()
mano_avg_gc2     = mvar_avg_gc2 - mclim_avg_gc2

mvar_avg_gc32 = ART1_gc32
mclim_avg_gc32    = mvar_avg_gc32.mean()
mano_avg_gc32     = mvar_avg_gc32 - mclim_avg_gc32

ob_s = mano_avg_jra**2
ob_sum = ob_s.sum()/nyrs

hkc_s_gc2 = (mano_avg_gc2 - mano_avg_jra)**2
hkc_sum_gc2 = hkc_s_gc2.sum()/nyrs
hkc_msss_gc2 = (1-(hkc_sum_gc2/ob_sum))

hkc_s_gc32 = (mano_avg_gc32 - mano_avg_jra)**2
hkc_sum_gc32 = hkc_s_gc32.sum()/nyrs
hkc_msss_gc32 = (1-(hkc_sum_gc32/ob_sum))

fhkc_msss_gc21  = "{:4.2f}".format(hkc_msss_gc2) 
fhkc_msss_gc321 = "{:4.2f}".format(hkc_msss_gc32) 



# ART2

mvar_avg_jra = ART2_jra
mclim_avg_jra    = mvar_avg_jra.mean()
mano_avg_jra     = mvar_avg_jra - mclim_avg_jra

mvar_avg_gc2 = ART2_gc2
mclim_avg_gc2    = mvar_avg_gc2.mean()
mano_avg_gc2     = mvar_avg_gc2 - mclim_avg_gc2

mvar_avg_gc32 = ART2_gc32
mclim_avg_gc32    = mvar_avg_gc32.mean()
mano_avg_gc32     = mvar_avg_gc32 - mclim_avg_gc32

ob_s = mano_avg_jra**2
ob_sum = ob_s.sum()/nyrs

hkc_s_gc2 = (mano_avg_gc2 - mano_avg_jra)**2
hkc_sum_gc2 = hkc_s_gc2.sum()/nyrs
hkc_msss_gc2 = (1-(hkc_sum_gc2/ob_sum))

hkc_s_gc32 = (mano_avg_gc32 - mano_avg_jra)**2
hkc_sum_gc32 = hkc_s_gc32.sum()/nyrs
hkc_msss_gc32 = (1-(hkc_sum_gc32/ob_sum))

fhkc_msss_gc22= "{:4.2f}".format(hkc_msss_gc2)
fhkc_msss_gc322="{:4.2f}".format(hkc_msss_gc32)

dcor_art1_gc2 = NCL_FUNC.escorc(dt_art1_jra,dt_art1_gc2)
dcor_art2_gc2 = NCL_FUNC.escorc(dt_art2_jra,dt_art2_gc2)


dcor_art1_gc32 =  NCL_FUNC.escorc(dt_art1_jra,dt_art1_gc32)
dcor_art2_gc32 =  NCL_FUNC.escorc(dt_art2_jra,dt_art2_gc32)

cor_art1_gc2 =  NCL_FUNC.escorc(ART1_jra,ART1_gc2)
cor_art2_gc2 =  NCL_FUNC.escorc(ART2_jra,ART2_gc2)


cor_art1_gc32 =  NCL_FUNC.escorc(ART1_jra,ART1_gc32)
cor_art2_gc32 =  NCL_FUNC.escorc(ART2_jra,ART2_gc32)

rmse_art1_gc2= NCL_FUNC.dim_rmsd_n(ART1_jra,ART1_gc2)
rmse_art1_gc32= NCL_FUNC.dim_rmsd_n(ART1_jra,ART1_gc32)

nrmse_art1_gc2 = rmse_art1_gc2/(ART1_jra.max()-ART1_jra.min())
nrmse_art1_gc32 = rmse_art1_gc32/(ART1_jra.max()-ART1_jra.min())

rmse_art2_gc2= NCL_FUNC.dim_rmsd_n(ART2_jra,ART2_gc2)
rmse_art2_gc32= NCL_FUNC.dim_rmsd_n(ART2_jra,ART2_gc32)

nrmse_art2_gc2 = rmse_art2_gc2/(ART2_jra.max()-ART2_jra.min())
nrmse_art2_gc32 = rmse_art2_gc32/(ART2_jra.max()-ART2_jra.min())


# dtrend rmse

drmse_art1_gc2= NCL_FUNC.dim_rmsd_n(dt_art1_jra,  dt_art1_gc2)
drmse_art1_gc32= NCL_FUNC.dim_rmsd_n(dt_art1_jra, dt_art1_gc32)

dnrmse_art1_gc2  = drmse_art1_gc2/(dt_art1_jra.max() - dt_art1_jra.min())
dnrmse_art1_gc32 = drmse_art1_gc32/(dt_art1_jra.max() - dt_art1_jra.min())

drmse_art2_gc2= NCL_FUNC.dim_rmsd_n(dt_art2_jra,  dt_art2_gc2)
drmse_art2_gc32= NCL_FUNC.dim_rmsd_n(dt_art2_jra, dt_art2_gc32)

dnrmse_art2_gc2  = drmse_art2_gc2/(dt_art2_jra.max() - dt_art2_jra.min())
dnrmse_art2_gc32 = drmse_art2_gc32/(dt_art2_jra.max() - dt_art2_jra.min())


rc_art1_jra,_ =  NCL_FUNC.regCoef_n(dt_art1_jra,dt_ea_jra)
rc_art2_jra,_ =  NCL_FUNC.regCoef_n(dt_art2_jra,dt_na_jra)
rc_art1_gc2,_ =  NCL_FUNC.regCoef_n(dt_art1_gc2,dt_ea_gc2)
rc_art2_gc2,_ =  NCL_FUNC.regCoef_n(dt_art2_gc2,dt_na_gc2)
rc_art1_gc32,_ =  NCL_FUNC.regCoef_n(dt_art1_gc32,dt_ea_gc32)
rc_art2_gc32,_ =  NCL_FUNC.regCoef_n(dt_art2_gc32,dt_na_gc32)



COR_ART1_SAT = [add_cyclic(COR_ART1_SAT_jra),add_cyclic(COR_ART1_SAT_gc2),add_cyclic(COR_ART1_SAT_gc32)]
COR_ART2_SAT = [add_cyclic(COR_ART2_SAT_jra),add_cyclic(COR_ART2_SAT_gc2),add_cyclic(COR_ART2_SAT_gc32)]
smap_art1    = [add_cyclic(smap_art1_jra),add_cyclic(smap_art1_gc2),add_cyclic(smap_art1_gc32)]
smap_art2    = [add_cyclic(smap_art2_jra),add_cyclic(smap_art2_gc2),add_cyclic(smap_art2_gc32)]
ART1_reg     = [ART1_jra ,ART1_gc2 ,ART1_gc32]
ART2_reg     = [ART2_jra ,ART2_gc2 ,ART2_gc32]
ART_reg      = [ART1_reg,ART2_reg]


rc_art1     = [rc_art1_jra, rc_art1_gc2, rc_art1_gc32]
rc_art2     = [rc_art2_jra, rc_art2_gc2, rc_art2_gc32]
pcc_art1    = [0.11111,pcc_art1_gc2, pcc_art1_gc32]
pcc_art2    = [0.11111,pcc_art2_gc2, pcc_art2_gc32]
nrmse_map1  = [0.11111,nrmse_map1_gc2, nrmse_map1_gc32]
nrmse_map2  = [0.11111,nrmse_map2_gc2, nrmse_map2_gc32]

cor_gc2     = [cor_art1_gc2[0],cor_art2_gc2[0]]
cor_gc32    = [cor_art1_gc32[0],cor_art2_gc32[0]]
dcor_gc2    = [dcor_art1_gc2[0],dcor_art2_gc2[0]]
dcor_gc32   = [dcor_art1_gc32[0],dcor_art2_gc32[0]]
nrmse_gc2   = [nrmse_art1_gc2.data, nrmse_art2_gc2.data]
nrmse_gc32  = [nrmse_art1_gc32.data, nrmse_art2_gc32.data]
dnrmse_gc2  = [dnrmse_art1_gc2.data, dnrmse_art2_gc2.data]
dnrmse_gc32 = [dnrmse_art1_gc32.data, dnrmse_art2_gc32.data]


latS1    = 70.0
latN1    = 80.0
lonL1    = 30.0
lonR1    = 70.0

latS11    = 35.0
latN11    = 80.0
lonL11    = 30.0
lonR11    = 130.0

latS2    = 65.0
latN2    = 80.0
lonL2    = 160.0
lonR2    = 200.0

latS22    = 35.0
latN22    = 80.0
lonL22    = 160.0
lonR22    = 280.0

art1_hdom = polar_proj_polyline(latS1,latN1,lonL1,lonR1)
art1_mdom = polar_proj_polyline(latS11,latN11,lonL11,lonR11)
art2_hdom = polar_proj_polyline(latS2,latN2,lonL2,lonR2)
art2_mdom = polar_proj_polyline(latS22,latN22,lonL22,lonR22)

syr, eyr=1993,2015

fig = plt.figure(figsize=(10, 10),dpi=300)

grid = gridspec.GridSpec(nrows=3, ncols=6, figure=fig,hspace=0.4, wspace= 0.5)
proj=projection=ccrs.NorthPolarStereo()


ax1 = fig.add_subplot(grid[0,:3])
ax2 = fig.add_subplot(grid[0,3:])
ax3 = fig.add_subplot(grid[1,:2], projection=proj)
ax4 = fig.add_subplot(grid[1,2:4], projection=proj)
ax5 = fig.add_subplot(grid[1,4:], projection=proj)
ax6 = fig.add_subplot(grid[2,:2], projection=proj)
ax7 = fig.add_subplot(grid[2,2:4], projection=proj)
ax8 = fig.add_subplot(grid[2,4:], projection=proj)

xy_axs=[ax1,ax2]
axs=[ax3,ax4,ax5,ax6,ax7,ax8]

for i in [0,1]:
    time = np.arange(syr, eyr + 1)
    xy_axs[i].plot(time, ART_reg[i][0], label='HadISST', color='black', marker='o')
    xy_axs[i].plot(time, ART_reg[i][1], color='red', marker='o')
    xy_axs[i].plot(time, ART_reg[i][2], color='blue', marker='o')
    xy_axs[i].axhline(y=0, color='black', linestyle='-', linewidth=0.6)


    xy_axs[i].grid(True, which='major', linestyle= (0,(5,3,5,10)), linewidth=0.6)
    xy_axs[i].set_xlabel('Year', fontsize=8)  # Corrected method

    gv.set_titles_and_labels(xy_axs[i],
                            lefttitle=f'({chr(ord("a")+i)}) ART{i+1} Index',
                            righttitle=f'[{season}, {syr}/{syr+1}-{eyr}/{eyr+1}]',
                            lefttitlefontsize=10,
                            righttitlefontsize=10)

    xy_axs[i].legend(labels=[
        'JRA55',
        f'GloSea5 (R={cor_gc2[i]:.2f}({dcor_gc2[i]:.2f})/NRMSE={nrmse_gc2[i]:.2f}({dnrmse_gc2[i]:.2f}))',
        f'GloSea6 (R={cor_gc32[i]:.2f}({dcor_gc32[i]:.2f})/NRMSE={nrmse_gc32[i]:.2f}({dnrmse_gc2[i]:.2f}))'
    ], fontsize=5, frameon=False)

    # Adjusting the tick parameters
    xy_axs[i].set_xlim([min(time),max(time)])
    xy_axs[i].set_ylim(-7,7)

    # xy_axs[i].set_yticks(np.arange(-4,5,2))
    xy_axs[i].set_xticks(np.arange(syr, eyr,4))


for i in range(len(axs)):
    axs[i].coastlines(resolution='10m', color='black', linewidth=0.3)
    axs[i].add_feature(cfeature.LAKES, edgecolor='black', linewidth=0.3, facecolor='none')

    gv.set_map_boundary(axs[i], [-180, 180], [30, 90], south_pad=1)

    gl = axs[i].gridlines(ccrs.PlateCarree(),
                    draw_labels=False,
                    linestyle=(0,(1,2)),
                    linewidth=0.5,
                    color='black',
                    zorder=2)

    gl.ylocator = mticker.FixedLocator(np.arange(30, 90, 15))
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 30))

    ticks = np.arange(0, 210, 30)
    etick = ['0'] + [f'{tick}E' for tick in ticks if (tick != 0) & (tick != 180)] + ['180']
    wtick = [f'{tick}W' % tick for tick in ticks if (tick != 0) & (tick != 180)]
    labels = [
        '0', '30E', '60E', '90E', '120E', '150E', '180', '150W', '120W',
        '90W', '60W', '30W'
    ]
    xticks = np.arange(0, 360, 30)
    yticks = np.full_like(xticks, 25)  # Latitude where the labels will be drawn

    tick_size=5
    for xtick, ytick, label in zip(xticks, yticks, labels):
        if label == '180':
            axs[i].text(xtick,
                    ytick,
                    label,
                    fontsize=tick_size,
                    horizontalalignment='center',
                    verticalalignment='top',
                    transform=ccrs.Geodetic())
        elif label == '0':
            axs[i].text(xtick,
                    ytick,
                    label,
                    fontsize=tick_size,
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    transform=ccrs.Geodetic())
        else:
            axs[i].text(xtick,
                    ytick,
                    label,
                    fontsize=tick_size,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ccrs.Geodetic())
    fig_n=['c','d','e','f','g','h']
    model_n=['JRA55', 'Glosea5','Glosea6']
    title_size=7
    v=np.arange(-0.6,0.7,0.1)
    cmap=cmaps.temp_diff_18lev
    significance_threshold = 0.49
    density=4
    if i <=2:
        C=axs[i].contourf(
            COR_ART1_SAT[i]['longitude'].data,
            COR_ART1_SAT[i]['latitude'].data,
            COR_ART1_SAT[i].data,
            transform=ccrs.PlateCarree(),
            levels=v,
            cmap=cmap,
            extend='both'
            )
        axs[i].contourf(smap_art1[i]['longitude'].data,
            smap_art1[i]['latitude'].data,
            smap_art1[i].data,
            transform=ccrs.PlateCarree(),
            colors='none',
            levels=[0,significance_threshold],
            hatches=[density*'.'],)
    
        gv.set_titles_and_labels(axs[i],
            lefttitle=f'({fig_n[i]}) SAT on ART1: {model_n[i]}',
            righttitle=f'[PCC={pcc_art1[i]:.2f}\nNRMSE={nrmse_map1[i]:.2f}]',
            lefttitlefontsize=title_size,
            righttitlefontsize=title_size)
        for j in range(4):
            axs[i].plot(art1_hdom[j][:, 0], art1_hdom[j][:, 1], color='lime', transform=ccrs.Geodetic())
            axs[i].plot(art1_mdom[j][:, 0], art1_mdom[j][:, 1], color='lime', transform=ccrs.Geodetic())
    else:
        axs[i].contourf(
            COR_ART2_SAT[i-3]['longitude'].data,
            COR_ART2_SAT[i-3]['latitude'].data,
            COR_ART2_SAT[i-3].data,
            transform=ccrs.PlateCarree(),
            levels=v,
            cmap=cmap,
            extend='both'
            )
        axs[i].contourf(smap_art2[i-3]['longitude'].data,
            smap_art2[i-3]['latitude'].data,
            smap_art2[i-3].data,
            transform=ccrs.PlateCarree(),
            colors='none',
            levels=[0,significance_threshold],
            hatches=[density*'.'],)
        gv.set_titles_and_labels(axs[i],
            lefttitle=f'({fig_n[i]}) SAT on ART2: {model_n[i-3]}',
            righttitle=f'[PCC={pcc_art2[i-3]:.2f}\nNRMSE={nrmse_map2[i-3]:.2f}]',
            lefttitlefontsize=title_size,
            righttitlefontsize=title_size)
        for j in range(4):
            axs[i].plot(art2_hdom[j][:, 0], art2_hdom[j][:, 1], color='lime', transform=ccrs.Geodetic())
            axs[i].plot(art2_mdom[j][:, 0], art2_mdom[j][:, 1], color='lime', transform=ccrs.Geodetic())
gv.set_titles_and_labels(axs[0],
                        righttitle=f' ')
gv.set_titles_and_labels(axs[3],
                        righttitle=f' ')
cax = plt.axes([0.25, 0.06, 0.53, 0.02])
v=np.arange(-0.5,0.6,0.1)
cbar=plt.colorbar(
    C,
    cax=cax,
    orientation='horizontal',
    ticks=v,
    extend='both'
)
ofile=f"./Figure/2-4.ART_IDX_{season}.png"
plt.savefig(ofile, bbox_inches='tight')
plt.close()