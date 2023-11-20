import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.contour as mcontour
from PIL import Image
import cartopy
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

season = "FM"

indir   = "../data/"
in_data = xr.open_dataset(f"{indir}DATA_ART_{season}.nc")
data_vars        = list(in_data.data_vars)

for var in data_vars[:]:
    globals()[var]=in_data[data_vars.pop(0)]
    
# COR_ART1_SAT_jra  = in_data[data_vars.pop(0)]
# COR_ART1_SAT_gc2  = in_data[data_vars.pop(0)]
# COR_ART1_SAT_gc32 = in_data[data_vars.pop(0)]
# COR_ART2_SAT_jra  = in_data[data_vars.pop(0)]
# COR_ART2_SAT_gc2  = in_data[data_vars.pop(0)]
# COR_ART2_SAT_gc32 = in_data[data_vars.pop(0)]
# smap_art1_jra     = in_data[data_vars.pop(0)]
# smap_art1_gc2     = in_data[data_vars.pop(0)]
# smap_art1_gc32    = in_data[data_vars.pop(0)]
# smap_art2_jra     = in_data[data_vars.pop(0)]
# smap_art2_gc2     = in_data[data_vars.pop(0)]
# smap_art2_gc32    = in_data[data_vars.pop(0)]
# ART1_jra          = in_data[data_vars.pop(0)]
# ART1_gc2          = in_data[data_vars.pop(0)]
# ART1_gc32         = in_data[data_vars.pop(0)]
# ART2_jra          = in_data[data_vars.pop(0)]
# ART2_gc2          = in_data[data_vars.pop(0)]
# ART2_gc32         = in_data[data_vars.pop(0)]

COR_ART1_SAT = [add_cyclic(COR_ART1_SAT_jra),add_cyclic(COR_ART1_SAT_gc2),add_cyclic(COR_ART1_SAT_gc32)]
COR_ART2_SAT = [add_cyclic(COR_ART2_SAT_jra),add_cyclic(COR_ART2_SAT_gc2),add_cyclic(COR_ART2_SAT_gc32)]
smap_art1    = [add_cyclic(smap_art1_jra),add_cyclic(smap_art1_gc2),add_cyclic(smap_art1_gc32)]
smap_art2    = [add_cyclic(smap_art2_jra),add_cyclic(smap_art2_gc2),add_cyclic(smap_art2_gc32)]
ART1_reg     = [ART1_jra ,ART1_gc2 ,ART1_gc32]
ART2_reg     = [ART2_jra ,ART2_gc2 ,ART2_gc32]
ART_reg      = [ART1_reg,ART2_reg]

cor_gc2     = [0.11111,0.22222]
dcor_gc2    = [0.11111,0.22222]
nrmse_gc2   = [0.11111,0.22222]
dnrmse_gc2  = [0.11111,0.22222]
cor_gc32    = [0.11111,0.22222]
dcor_gc32   = [0.11111,0.22222]
nrmse_gc32  = [0.11111,0.22222]
dnrmse_gc32 = [0.11111,0.22222]
rc_art1     = [0.11111,0.22222,0.33333]
rc_art2     = [0.11111,0.22222,0.33333]
pcc_art1    = [0.11111,0.22222,0.33333]
pcc_art2    = [0.11111,0.22222,0.33333]
nrmse_map1  = [0.11111,0.22222,0.33333]
nrmse_map2  = [0.11111,0.22222,0.33333]

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
        f'GloSea6 (R={cor_gc32[i]:.2f}({dcor_gc32[i]:.2f})/NRMSE={nrmse_gc32[i]:.2f}({dnrmse_gc32[i]:.2f}))'
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