import numpy as np
import xarray as xr
from sklearn.metrics import mean_squared_error
import sys, os
sys.path.append("/data03/Glosea5/jhsim/NCL2PYTHON/METRIC_PYTHON/src")
import READ_FILE
import NCL_FUNC

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

# READ GloSea5
model0 = "GloSea5"
tvar = "hgt"
tlev = 500
mz5003_gc2 = READ_FILE.READ_GloSea5(season,tvar,tlev,model0)

tlev = 1000
mz10003_gc2 = READ_FILE.READ_GloSea5(season,tvar,tlev,model0)

tvar = "t15m"
mt2m3_gc2 = READ_FILE.READ_GloSea5(season,tvar,tlev,model0)

# READ GloSea6
model0 = "GloSea6"
tvar = "hgt"
tlev = 500
mz5003_gc32 = READ_FILE.READ_GloSea5(season,tvar,tlev,model0)

tlev = 1000
mz10003_gc32 = READ_FILE.READ_GloSea5(season,tvar,tlev,model0)

tvar = "t15m"
mt2m3_gc32 = READ_FILE.READ_GloSea5(season,tvar,tlev,model0)


oz5003.name       ="JRA_z500"
oz10003.name      ="JRA_z1000"
ot2m3.name        ="JRA_t2m"
mz5003_gc2.name   ="GloSea5_z500"
mz10003_gc2.name  ="GloSea5_z1000"
mt2m3_gc2.name    ="GloSea5_t2m"
mz5003_gc32.name  ="GloSea6_z500"
mz10003_gc32.name ="GloSea6_z1000"
mt2m3_gc32.name   ="GloSea6_t2m"

datalist=[]


jra=xr.concat([oz5003,oz10003,ot2m3],dim='JRA')
jra.
# mz5003_gc2m,mz10003_gc2m,mt2m3_gc2
# mz5003_gc32,mz10003_gc32,mt2m3_gc32



