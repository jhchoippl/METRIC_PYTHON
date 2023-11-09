from wrf import omp_set_num_threads, omp_get_num_procs
import numpy as np
import time

num_threads=omp_get_num_procs()

import numpy as np
import xarray as xr
from sklearn.metrics import mean_squared_error
import sys, os
sys.path.append("../src")
import READ_FILE
import NCL_FUNC

import cmaps
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geocat.viz as gv

season = "ON"

# READ JRA55
tvar = "hgt"
tlev = 500
oz5003=READ_FILE.READ_JRA55(season,tvar,tlev)

tlev = 1000
oz10003 = READ_FILE.READ_JRA55(season,tvar,tlev)

tvar = "tmp2m"
ot2m3 = READ_FILE.READ_JRA55(season,tvar,tlev)

nyrs=oz5003['year'].size
owgt=NCL_FUNC.latRegWgt(oz5003['latitude'])
wgt=owgt[owgt['latitude']>=20]
wgt.name='weights'

model=["GloSea5","GloSea6"]

# READ GloSea5
tvar = "hgt"
tlev = 500
mz5003_gc2 = READ_FILE.READ_GloSea5(season,tvar,tlev,model[0])

tlev = 1000
mz10003_gc2 = READ_FILE.READ_GloSea5(season,tvar,tlev,model[0])

tvar = "t15m"
mt2m3_gc2 = READ_FILE.READ_GloSea5(season,tvar,tlev,model[0])

# READ GloSea6
tvar = "hgt"
tlev = 500
mz5003_gc32 = READ_FILE.READ_GloSea5(season,tvar,tlev,model[1])

tlev = 1000
mz10003_gc32 = READ_FILE.READ_GloSea5(season,tvar,tlev,model[1])

tvar = "t15m"
mt2m3_gc32 = READ_FILE.READ_GloSea5(season,tvar,tlev,model[1])