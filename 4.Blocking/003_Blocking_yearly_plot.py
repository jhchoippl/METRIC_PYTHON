import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import geocat.viz as gv

season = "ON"

indir   = "../data/"
in_data = xr.open_dataset(f"{indir}DATA_Blocking_{season}.nc")
data_vars        = list(in_data.data_vars)
for var in data_vars[:]:
    globals()[var]=in_data[data_vars.pop(0)]
    # fyyindex_ub_jra
    # fyyindex_kb_jra
    # fyyindex_esb_jra
    # fyyindex_ub_gc2
    # fyyindex_kb_gc2
    # fyyindex_esb_gc2
    # fyyindex_ub_gc32
    # fyyindex_kb_gc32
    # fyyindex_esb_gc32
fyyindex_jra  = [fyyindex_ub_jra,fyyindex_esb_jra,fyyindex_kb_jra]
fyyindex_gc2  = [fyyindex_ub_gc2,fyyindex_esb_gc2,fyyindex_kb_gc2]
fyyindex_gc32 = [fyyindex_ub_gc32,fyyindex_esb_gc32,fyyindex_kb_gc32]
cor_gc2       = [0.11111,0.22222,0.33333]
cor_gc32      = [0.11111,0.22222,0.33333]
nrmse_gc2     = [0.11111,0.22222,0.33333]
nrmse_gc32    = [0.11111,0.22222,0.33333]

syr, eyr=1993,2015

fig = plt.figure(figsize=(10, 10),dpi=300)

grid = gridspec.GridSpec(nrows=3, ncols=1, figure=fig,hspace=0.4, wspace= 0.5)
ax1 = fig.add_subplot(grid[0,:])
ax2 = fig.add_subplot(grid[1,:])
ax3 = fig.add_subplot(grid[2,:])

axs=[ax1,ax2,ax3]
time = np.arange(syr, eyr + 1)
blk_n=['Ural','East Siberian','Kamchatka']
for i in [0,1,2]:
    axs[i].plot(time, fyyindex_jra[i], label='HadISST', color='black', marker='o')
    axs[i].plot(time, fyyindex_gc2[i], color='red', marker='o')
    axs[i].plot(time, fyyindex_gc32[i], color='blue', marker='o')

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
        f'GloSea5 (R={cor_gc2[i]:.2f}/NRMSE={nrmse_gc2[i]:.2f}',
        f'GloSea6 (R={cor_gc32[i]:.2f}/NRMSE={nrmse_gc32[i]:.2f}'
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