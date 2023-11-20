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

season = "FM"
syr, eyr=1993,2015

indir            = "../data/"
in_data          = xr.open_dataset(f"{indir}DATA_AO_{season}.nc")
data_vars        = list(in_data.data_vars)

eof_regres_jra=in_data[data_vars.pop(0)]
eof_ts_jra=in_data[data_vars.pop(0)]
eof_ts_gc2=in_data[data_vars.pop(0)]
eof_ts_gc32=in_data[data_vars.pop(0)]
cor_gc2          = 0.11111
cor_gc32         = 0.11111
dcor_gc2         = 0.11111
dcor_gc32        = 0.11111
nrmse_gc2        = 0.11111
nrmse_gc32       = 0.11111
dnrmse_gc2       = 0.11111
dnrmse_gc32      = 0.11111

eof_regres_jra=eof_regres_jra.rename({'lon':'longitude','lat':'latitude'})
eof_regres_jra=eof_regres_jra.drop_vars('evn').squeeze('evn')
eof_regres_jra=add_cyclic(eof_regres_jra)

eof_ts=[eof_ts_jra,eof_ts_gc2,eof_ts_gc32]









fig = plt.figure(figsize=(10, 10),dpi=300)

grid = gridspec.GridSpec(nrows=3, ncols=2, figure=fig,hspace=0.4, wspace=-0.5)
proj=projection=ccrs.NorthPolarStereo()

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
                        righttitle=f'[{season}, {syr}/{syr+1}-{eyr}/{eyr+1}]',
                        lefttitlefontsize=12,
                        righttitlefontsize=12)

ax2.legend(labels=[
    'JRA55',
    f'GloSea5 (R={cor_gc2:.2f}({dcor_gc2:.2f})/NRMSE={nrmse_gc2:.2f}({dnrmse_gc2:.2f}))',
    f'GloSea6 (R={cor_gc32:.2f}({dcor_gc32:.2f})/NRMSE={nrmse_gc32:.2f}({dnrmse_gc32:.2f}))'
], fontsize=8, frameon=False)

# Adjusting the tick parameters
ax2.set_xlim([min(time),max(time)])
ax2.set_ylim(-3,4)

# ax2.set_yticks(np.arange(-4,5,2))
ax2.set_xticks(np.arange(syr, eyr,4))

ofile=f"./Figure/2-1_AO_{season}.png"
plt.savefig(ofile, bbox_inches='tight')
plt.close()