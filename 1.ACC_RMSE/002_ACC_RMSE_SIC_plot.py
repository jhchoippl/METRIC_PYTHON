import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.contour as mcontour
from PIL import Image
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import cmaps
import geocat.viz as gv

def add_cyclic(data):
    import geocat.viz as gv
    return gv.xr_add_cyclic_longitudes(data,'longitude')

season = "DJ"

indir            = "../data/"
in_data          = xr.open_dataset(f"{indir}DATA_002_{season}.nc")
data_vars        = list(in_data.data_vars)
sic3_acc_gc2     = in_data[data_vars.pop(0)]
sic3_acc_gc32    = in_data[data_vars.pop(0)]
tkc3_smap_gc2    = in_data[data_vars.pop(0)]
tkc3_smap_gc32   = in_data[data_vars.pop(0)]
sic3_rmse_gc2    = in_data[data_vars.pop(0)]
sic3_rmse_gc32   = in_data[data_vars.pop(0)]
msic3_cdiff_gc2  = in_data[data_vars.pop(0)]
msic3_cdiff_gc32 = in_data[data_vars.pop(0)]
reg_osic         = in_data[data_vars.pop(0)]
reg_msic_gc2     = in_data[data_vars.pop(0)]
reg_msic_gc32    = in_data[data_vars.pop(0)]
cor_gc2          = 0.11111
cor_gc32         = 0.11111
dcor_gc2         = 0.11111
dcor_gc32        = 0.11111
nrmse_gc2        = 0.11111
nrmse_gc32       = 0.11111
dnrmse_gc2       = 0.11111
dnrmse_gc32      = 0.11111
c_acc_gc2        = 0.11111
c_acc_gc32       = 0.11111
c_rmse_gc2       = 0.11111
c_rmse_gc32      = 0.11111

sic3_acc    = [add_cyclic(sic3_acc_gc2),add_cyclic(sic3_acc_gc32)]
tkc3_smap   = [add_cyclic(tkc3_smap_gc2),add_cyclic(tkc3_smap_gc32)]
sic3_rmse   = [add_cyclic(sic3_rmse_gc2),add_cyclic(sic3_rmse_gc32)]
msic3_cdiff = [add_cyclic(msic3_cdiff_gc2),add_cyclic(msic3_cdiff_gc32)]

data=[msic3_cdiff,sic3_acc,sic3_rmse]

reg_sic    = [reg_osic,reg_msic_gc2,reg_msic_gc32]

syr, eyr=1993,2015

time = np.arange(syr, eyr + 1)
xy_plot_y_lim=[-5,5]

fig = plt.figure(figsize=(10, 4),dpi=300)
ax=plt.axes()

ax.plot(time, reg_sic[0], label='HadISST', color='black', marker='o')
ax.plot(time, reg_sic[1], color='red', marker='o')
ax.plot(time, reg_sic[2], color='blue', marker='o')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.6)


ax.grid(True, which='major', linestyle= (0,(5,3,5,10)), linewidth=0.6)
ax.set_xlabel('Year', fontsize=16)  # Corrected method
ax.set_ylabel('[%]', fontsize=16)  # Corrected metho

gv.set_titles_and_labels(ax,
                        lefttitle=f'SIC Anomaly',
                        righttitle=f'[{season}, {syr}/{syr+1}-{eyr}/{eyr+1}]',
                        lefttitlefontsize=12,
                        righttitlefontsize=12)

ax.legend(labels=[
    'JRA55',
    f'GloSea5 (R={cor_gc2:.2f}({dcor_gc2:.2f})/NRMSE={nrmse_gc2:.2f}({dnrmse_gc2:.2f}))',
    f'GloSea6 (R={cor_gc32:.2f}({dcor_gc32:.2f})/NRMSE={nrmse_gc32:.2f}({dnrmse_gc32:.2f}))'
], fontsize=8, frameon=False)

# Adjusting the tick parameters
ax.set_xlim([min(time),max(time)])
ax.set_ylim(xy_plot_y_lim)

# ax.set_yticks(np.arange(-4,5,2))
ax.set_xticks(np.arange(syr, eyr,3))

ofile1=f"./Figure/1-1_ACC_RMSE_SIC_{season}_xy.png"
plt.savefig(ofile1, bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(5, 5),dpi=300)

grid = gridspec.GridSpec(nrows=3, ncols=2, figure=fig,hspace=0.4, wspace=-0.5)
proj=projection=ccrs.NorthPolarStereo()

ax1 = fig.add_subplot(grid[0,0], projection=proj)  # middle cell of grid
ax2 = fig.add_subplot(grid[0,1], projection=proj)  # middle cell of grid
ax3 = fig.add_subplot(grid[1,0], projection=proj)  # middle cell of grid
ax4 = fig.add_subplot(grid[1,1], projection=proj)  # middle cell of grid
ax5 = fig.add_subplot(grid[2,0], projection=proj)  # middle cell of grid
ax6 = fig.add_subplot(grid[2,1], projection=proj)  # middle cell of grid

axs=[[ax1,ax2],[ax3,ax4],[ax5,ax6]]
metric        = ["Mean Bias", "ACC", 'RMSE']
title         = ["Glosea5 minus HadISST","Glosea6 minus HadISST","Glosea5","Glosea6","Glosea5","Glosea6"]



right_title= ["","",c_acc_gc2,c_acc_gc32,c_rmse_gc2,c_rmse_gc32]


for row in range(0, 3):
    for col in range(0, 2):
        axs[row][col].coastlines(resolution='10m', color='black', linewidth=0.1)
        axs[row][col].add_feature(cfeature.LAND, facecolor='lightgray')
        axs[row][col].add_feature(cfeature.LAKES, edgecolor='black', linewidth=0.1, facecolor='none')

        gv.set_map_boundary(axs[row][col], [-180, 180], [50, 90], south_pad=1)

        gl = axs[row][col].gridlines(ccrs.PlateCarree(),
                        draw_labels=False,
                        linestyle=(0,(1,2)),
                        linewidth=0.5,
                        color='black',
                        zorder=2)

        gl.ylocator = mticker.FixedLocator(np.arange(0, 90, 10))
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 60))

        ticks = np.arange(0, 210, 30)
        etick = ['0'] + [f'{tick}E' for tick in ticks if (tick != 0) & (tick != 180)] + ['180']
        wtick = [f'{tick}W' % tick for tick in ticks if (tick != 0) & (tick != 180)]
        labels = [
            '0', '30E', '60E', '90E', '120E', '150E', '180', '150W', '120W',
            '90W', '60W', '30W'
        ]
        xticks = np.arange(0, 360, 30)
        yticks = np.full_like(xticks, 43)  # Latitude where the labels will be drawn

        tick_size=3
        for xtick, ytick, label in zip(xticks, yticks, labels):
            if label == '180':
                axs[row][col].text(xtick,
                        ytick,
                        label,
                        fontsize=tick_size,
                        horizontalalignment='center',
                        verticalalignment='top',
                        transform=ccrs.Geodetic())
            elif label == '0':
                axs[row][col].text(xtick,
                        ytick,
                        label,
                        fontsize=tick_size,
                        horizontalalignment='center',
                        verticalalignment='bottom',
                        transform=ccrs.Geodetic())
            else:
                axs[row][col].text(xtick,
                        ytick,
                        label,
                        fontsize=tick_size,
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ccrs.Geodetic())
        
        title_size=3.5
        if row==0:
            gv.set_titles_and_labels(axs[row][col],
                                    lefttitle=f'{metric[row]}({title[(row*2)+col]})',
                                    righttitle='',
                                    lefttitlefontsize=title_size,
                                    righttitlefontsize=title_size)
        else :
        # Use geocat.viz.util convenience function to set titles
            gv.set_titles_and_labels(axs[row][col],
                                    lefttitle=f'{metric[row]}({title[(row*2)+col]})',
                                    righttitle=f'[{metric[row]}: {right_title[(row*2)+col]:.2f}]', 
                                    lefttitlefontsize=title_size,
                                    righttitlefontsize=title_size)
        if row==0:
            cmap=cmaps.BlueWhiteOrangeRed
            v=list(range(-45,46,5))
            extend='both'
        elif row ==1:
            cmap=cmaps.temp_diff_18lev
            v= np.linspace(-1, 1, 21)
            extend='neither'
        elif row==2:
            cmap=cmaps.WhiteBlueGreenYellowRed
            v= np.arange(0, 51, 5)
            extend='both'
            
        if col==0:
            C=axs[row][col].contourf(
                data[row][col]['longitude'].data,
                data[row][col]['latitude'].data,
                data[row][col].data,
                transform=ccrs.PlateCarree(),
                levels=v,
                cmap=cmap,
                extend=extend
                )
            
        elif col ==1:      
            axs[row][col].contourf(
                data[row][col]['longitude'].data,
                data[row][col]['latitude'].data,
                data[row][col].data,
                transform=ccrs.PlateCarree(),
                levels=v,
                cmap=cmap,
                extend=extend
                )
            
            if row==0:
                cmap=cmaps.BlueWhiteOrangeRed
                v=list(range(-45,46,15))
            elif row ==1:
                cmap=cmaps.temp_diff_18lev
                v= np.linspace(-0.9, 0.9, 7)
            elif row==2:
                cmap=cmaps.WhiteBlueGreenYellowRed
                v= np.arange(0, 51, 5)
            cbar = plt.colorbar(C,
                                ax=axs[row][:],
                                ticks=v,
                                extendfrac='auto',
                                aspect=15,
                                orientation='horizontal',
                                extend='both',
                                drawedges=True,
                                fraction=0.07
                                )
            cbar.ax.tick_params(labelsize=4)
significance_threshold = 0.49
density=5
for i in [0,1]:
    axs[1][i].contourf(tkc3_smap[i]['longitude'].data,
        tkc3_smap[i]['latitude'].data,
        tkc3_smap[i].data,
        transform=ccrs.PlateCarree(),
        colors='none',
        levels=[0,significance_threshold],
        hatches=[density*'.'],)
    
ofile2=f"./Figure/1-1_ACC_RMSE_SIC_{season}_polar.png"
plt.savefig(ofile2, bbox_inches='tight')
plt.close()


image1 = Image.open(ofile1)
image2 = Image.open(ofile2)

data1 = np.array(image1)
data2 = np.array(image2)

gs = gridspec.GridSpec(5, 2,hspace=0)

fig = plt.figure(figsize=(10, 10),dpi=300)
ax1=fig.add_subplot(gs[0,:])
ax1.imshow(data1)
ax1.axis('off')

ax2=fig.add_subplot(gs[1:,:])
ax2.imshow(data2)
ax2.axis('off')

plt.tight_layout(pad=0)

ofile3=f"./Figure/1-1_ACC_RMSE_SIC_{season}.png"
plt.savefig(ofile3, bbox_inches='tight')
plt.close()

os.remove(ofile1)
os.remove(ofile2)