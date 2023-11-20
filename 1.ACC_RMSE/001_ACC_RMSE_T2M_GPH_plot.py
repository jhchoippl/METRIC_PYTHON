import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import cmaps
import geocat.viz as gv

def PanelLabelBar(mappable,ax,ticks,aspect):
    import matplotlib.pyplot as plt
    plt.colorbar(mappable,
             ax=ax,
             ticks=ticks,
             extendfrac='auto',
             aspect=aspect,
             drawedges=True,
             orientation='horizontal',
             pad=0.2,
             fraction=0.125
             )
    
def add_cyclic(data):
    import geocat.viz as gv
    return gv.xr_add_cyclic_longitudes(data,'longitude')


season = "DJ"

indir            = "../data/"
data=xr.open_dataset(f"{indir}DATA_001_{season}.nc")

data_vars=list(data.data_vars)

t2m3_acc_gc2       = data[data_vars.pop(0)]
t2m3_acc_gc32      = data[data_vars.pop(0)]
z5003_acc_gc2      = data[data_vars.pop(0)]
z5003_acc_gc32     = data[data_vars.pop(0)]
z10003_acc_gc2     = data[data_vars.pop(0)]
z10003_acc_gc32    = data[data_vars.pop(0)]
tkc3_smap_gc2      = data[data_vars.pop(0)]
tkc3_smap_gc32     = data[data_vars.pop(0)]
hkc3_smap_gc2      = data[data_vars.pop(0)]
hkc3_smap_gc32     = data[data_vars.pop(0)]
hkc3_smap_gc22     = data[data_vars.pop(0)]
hkc3_smap_gc322    = data[data_vars.pop(0)]
mt2m3_cdiff_gc2    = data[data_vars.pop(0)]
mt2m3_cdiff_gc32   = data[data_vars.pop(0)]
mz5003_cdiff_gc2   = data[data_vars.pop(0)]
mz5003_cdiff_gc32  = data[data_vars.pop(0)]
mz10003_cdiff_gc2  = data[data_vars.pop(0)]
mz10003_cdiff_gc32 = data[data_vars.pop(0)]
t2m3_rmse_gc2      = data[data_vars.pop(0)]
t2m3_rmse_gc32     = data[data_vars.pop(0)]
z5003_rmse_gc2     = data[data_vars.pop(0)]
z5003_rmse_gc32    = data[data_vars.pop(0)]
z10003_rmse_gc2    = data[data_vars.pop(0)]
z10003_rmse_gc32   = data[data_vars.pop(0)]
reg_ot2m           = data[data_vars.pop(0)]
reg_oz1000         = data[data_vars.pop(0)]
reg_oz500          = data[data_vars.pop(0)]
reg_mt2m_gc2       = data[data_vars.pop(0)]
reg_mt2m_gc32      = data[data_vars.pop(0)]
reg_mz1000_gc2     = data[data_vars.pop(0)]
reg_mz1000_gc32    = data[data_vars.pop(0)]
reg_mz500_gc2      = data[data_vars.pop(0)]
reg_mz500_gc32     = data[data_vars.pop(0)]

cdiff_gc2   = [add_cyclic(mt2m3_cdiff_gc2),add_cyclic(mz5003_cdiff_gc2),add_cyclic(mz10003_cdiff_gc2)]
cdiff_gc32  = [add_cyclic(mt2m3_cdiff_gc32),add_cyclic(mz5003_cdiff_gc32),add_cyclic(mz10003_cdiff_gc32)]
acc_gc2     = [add_cyclic(t2m3_acc_gc2),add_cyclic(z5003_acc_gc2),add_cyclic(z10003_acc_gc2)]
smap_gc2    = [add_cyclic(tkc3_smap_gc2),add_cyclic(hkc3_smap_gc2),add_cyclic(hkc3_smap_gc22)]
acc_gc32    = [add_cyclic(t2m3_acc_gc32),add_cyclic(z5003_acc_gc32),add_cyclic(z10003_acc_gc32)]
smap_gc32   = [add_cyclic(tkc3_smap_gc32),add_cyclic(hkc3_smap_gc32),add_cyclic(hkc3_smap_gc322)]
rmse_gc2    = [add_cyclic(t2m3_rmse_gc2),add_cyclic(z5003_rmse_gc2),add_cyclic(z10003_rmse_gc2)]
rmse_gc32   = [add_cyclic(t2m3_rmse_gc32),add_cyclic(z5003_rmse_gc32),add_cyclic(z10003_rmse_gc32)]
reg_jra     = [reg_ot2m,reg_oz500,reg_oz1000]
reg_gc2     = [reg_mt2m_gc2,reg_mz500_gc2,reg_mz1000_gc2]
reg_gc3     = [reg_mt2m_gc32,reg_mz500_gc32,reg_mz1000_gc32]
cor_gc2     = [0.11111,0.22222,0.33333]
cor_gc32    = [0.11111,0.22222,0.33333]
dcor_gc2    = [0.11111,0.22222,0.33333]
dcor_gc32   = [0.11111,0.22222,0.33333]
nrmse_gc2   = [0.11111,0.22222,0.33333]
nrmse_gc32  = [0.11111,0.22222,0.33333]
dnrmse_gc2  = [0.11111,0.22222,0.33333]
dnrmse_gc32 = [0.11111,0.22222,0.33333]
c_acc_gc2   = [0.11111,0.22222,0.33333]
c_acc_gc32  = [0.11111,0.22222,0.33333]
c_rmse_gc2  = [0.11111,0.22222,0.33333]
c_rmse_gc32 = [0.11111,0.22222,0.33333]



min_lat = 20
max_lat = 90
min_lon = 0
max_lon = 360

syr, eyr = [reg_ot2m.year.data[i] for i in [0, -1]]
time = np.arange(syr, eyr + 1)

vars          = ["T2M","Z500","Z1000"]
metric        = ["Mean Bias", "ACC", 'RMSE']
title         = ["Glosea5 minus JRA55","Glosea6 minus JRA55","JRA55,Glosea5","JRA55,Glosea6","JRA55,Glosea5","JRA55,Glosea6"]
xy_plot_y_lim = [[-1,1],[-15,15],[-10,10]]


for v_i in range(len(vars)):
    fig = plt.figure(figsize=(10, 10),dpi=300)

    grid = gridspec.GridSpec(nrows=4, ncols=2, figure=fig,hspace=0.4)

    # Choose the map projection
    proj = ccrs.PlateCarree(central_longitude=180)

    ax1 = fig.add_subplot(grid[0:1,0:2])  # upper cell of grid
    ax2 = fig.add_subplot(grid[1,0], projection=proj)  # middle cell of grid
    ax3 = fig.add_subplot(grid[1,1], projection=proj)  # lower cell of grid
    ax4 = fig.add_subplot(grid[2,0], projection=proj)  # middle cell of grid
    ax5 = fig.add_subplot(grid[2,1], projection=proj)  # lower cell of grid
    ax6 = fig.add_subplot(grid[3,0], projection=proj)  # middle cell of grid
    ax7 = fig.add_subplot(grid[3,1], projection=proj)  # lower cell of grid

    axs=[ax2, ax3, ax4, ax5, ax6, ax7]


    ax1.plot(time, reg_jra[v_i], label='JRA55', color='black', marker='o')
    ax1.plot(time, reg_gc2[v_i], color='red', marker='o')
    ax1.plot(time, reg_gc3[v_i], color='blue', marker='o')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.6)


    ax1.grid(True, which='major', linestyle= (0,(5,3,5,10)), linewidth=0.6)
    ax1.set_xlabel('Year', fontsize=16)  # Corrected method
    ax1.set_ylabel('[°C]', fontsize=16)  # Corrected metho

    gv.set_titles_and_labels(ax1,
                            lefttitle=f'{vars[v_i]} Anomaly',
                            righttitle=f'[{season}, {syr}/{syr+1}-{eyr-1}/{eyr}]',
                            lefttitlefontsize=12,
                            righttitlefontsize=12)

    ax1.legend(labels=[
        'JRA55',
        f'GloSea5 (R={cor_gc2[v_i]:.2f}({dcor_gc2[v_i]:.2f})/NRMSE={nrmse_gc2[v_i]:.2f}({dnrmse_gc2[v_i]:.2f}))',
        f'GloSea6 (R={cor_gc32[v_i]:.2f}({dcor_gc32[v_i]:.2f})/NRMSE={nrmse_gc32[v_i]:.2f}({dnrmse_gc32[v_i]:.2f}))'
    ], fontsize=8, frameon=False)

    # Adjusting the tick parameters
    ax1.set_xlim([min(time),max(time)])
    ax1.set_ylim(xy_plot_y_lim[v_i])
    ax1.set_xticks(np.arange(syr, eyr,4))

    ax1.minorticks_on()

    ax1.tick_params(axis = 'both', which='major', labelsize = 13, length=10)
    ax1.tick_params(axis = 'both', which='minor', length=4)

    right_title= ["","",c_acc_gc2[v_i],c_acc_gc32[v_i],c_rmse_gc2[v_i],c_rmse_gc32[v_i]]

    for i in range(len(axs)):
        ax=axs[i]
        # Use geocat.viz.util convenience function to set axes tick values
        ax.coastlines(resolution='10m', color='black', linewidth=0.1)
        ax.add_feature(cfeature.LAKES, edgecolor='black', linewidth=0.1, facecolor='none')
        gv.set_axes_limits_and_ticks(ax=ax,
                                    xlim=(-180, 180),
                                    ylim=(min_lat, max_lat),
                                    yticks=np.arange(min_lat, max_lat, 30),
                                    xticks=np.arange(-180, 181, 60))

        gv.add_lat_lon_ticklabels(ax)

        # Remove the degree symbol from tick labels
        ax.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol=''))
        ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol=''))

        # Use geocat.viz.util convenience function to set titles
        if i==0 or i==1:
            gv.set_titles_and_labels(ax,
                                    lefttitle=f'{vars[v_i]} {metric[i//2]}({title[i]})',
                                    righttitle='',
                                    lefttitlefontsize=10,
                                    righttitlefontsize=10)
        else :
        # Use geocat.viz.util convenience function to set titles
            gv.set_titles_and_labels(ax,
                                    lefttitle=f'{vars[v_i]} {metric[i//2]}({title[i]})',
                                    righttitle=f'[{metric[i//2]}: {right_title[i]:.2f}]', 
                                    lefttitlefontsize=10,
                                    righttitlefontsize=10)

    aspect=20
    cmap=cmaps.BlueWhiteOrangeRed
    if v_i==0:
        v= np.linspace(-6, 6, 13)
    else:
        v= np.linspace(-6, 6, 13)*10

    C = ax2.contourf(cdiff_gc2[v_i]['longitude'].data,
        cdiff_gc2[v_i]['latitude'].data,
        cdiff_gc2[v_i].data,
        transform=ccrs.PlateCarree(),
        levels=v,
        cmap=cmap,
        extend='both')

    ax3.contourf(cdiff_gc32[v_i]['longitude'].data,
        cdiff_gc32[v_i]['latitude'].data,
        cdiff_gc32[v_i].data,
        transform=ccrs.PlateCarree(),
        levels=v,
        cmap=cmap,
        extend='both')

    PanelLabelBar(C,[ax2, ax3],v,aspect-0.6)


    cmap=cmaps.temp_diff_18lev
    v= np.linspace(-1, 1, 21)

    C2 = ax4.contourf(acc_gc2[v_i]['longitude'].data,
        acc_gc2[v_i]['latitude'].data,
        acc_gc2[v_i].data,
        transform=ccrs.PlateCarree(),
        levels=v,
        cmap=cmap,
        extend='neither')

    lon=tkc3_smap_gc2['longitude']
    lat=tkc3_smap_gc2['latitude']

    lon2d, lat2d = np.meshgrid(lon, lat)
    thinning_factor=5

    significance_threshold = 0.49

    density=4

    ax4.contourf(smap_gc2[v_i]['longitude'].data,
        smap_gc2[v_i]['latitude'].data,
        smap_gc2[v_i].data,
        transform=ccrs.PlateCarree(),
        colors='none',
        levels=[0,significance_threshold],
        hatches=[density*'.'],)

    ax5.contourf(acc_gc32[v_i]['longitude'].data,
        acc_gc32[v_i]['latitude'].data,
        acc_gc32[v_i].data,
        transform=ccrs.PlateCarree(),
        levels=v,
        cmap=cmap,
        extend='neither')

    ax5.contourf(smap_gc32[v_i]['longitude'].data,
        smap_gc32[v_i]['latitude'].data,
        smap_gc32[v_i].data,
        transform=ccrs.PlateCarree(),
        colors='none',
        levels=[0,significance_threshold],
        hatches=[density*'.'],)


    v=np.linspace(-1, 1, 11)
    PanelLabelBar(C2,[ax4, ax5],v,aspect+2.2)

    cmap=cmaps.WhiteBlueGreenYellowRed
    if v_i==0:
        v= np.linspace(0, 10, 11)
    else:
        v= np.linspace(0, 10, 11)*10

    C3 = ax6.contourf(rmse_gc2[v_i]['longitude'].data,
        rmse_gc2[v_i]['latitude'].data,
        rmse_gc2[v_i].data,
        transform=ccrs.PlateCarree(),
        levels=v,
        cmap=cmap,
        extend='max')

    ax7.contourf(rmse_gc32[v_i]['longitude'].data,
        rmse_gc32[v_i]['latitude'].data,
        rmse_gc32[v_i].data,
        transform=ccrs.PlateCarree(),
        levels=v,
        cmap=cmap,
        extend='max')

    PanelLabelBar(C3,[ax6, ax7],v,aspect+0.5)
    ofile=f"./Figure/test_1-1_ACC_RMSE_T2M_GPH_{season}.{vars[v_i]}.png"
    plt.savefig(ofile, bbox_inches='tight')
    plt.close()