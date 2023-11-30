import xarray as xr
import numpy as np
import os
from datetime import datetime

model = "JRA55"

wdir = "/data03/Glosea5/jhsim/NCL2PYTHON/METRIC_NCL"
root = f"{wdir}/4.Blocking/{model}/"
ofile = f"/data03/Glosea5/jhsim/NCL2PYTHON/METRIC_PYTHON/4.Blocking/{model}"

idir = f"{root}/ofile/"
odir = f"{ofile}/index/"
os.makedirs(odir, exist_ok=True)

vname = "hgt"
for season in ["ON","DJ","FM"]:

    if season == "DJ":
        syr, eyr, init = 1993, 2015, "1201"
    elif season == "FM":
        syr, eyr, init = 1994, 2016, "0201"
    elif season == "ON":
        syr, eyr, init = 1993, 2015, "1001"

    for yy in range(syr, eyr + 1):
        ifname = f"{model}.IB.index.{yy}{init}"
        ofname = f"{model}.{yy}{init}"
        ifile_path = f"{idir}{ifname}.nc"

        ds = xr.open_dataset(ifile_path)
        time = ds['time'].values
        lon = ds['lon'].values
        var = ds[vname].values[:, 0, 0, :]

        var_new = np.zeros_like(var)

        for j in range(len(lon)):
            i = 0
            while i < len(time) - 1:
                if var[i, j] == 1:
                    cnt = 1
                    while (i + cnt < len(time) - 1 and var[i + cnt, j] == 1 and
                        datetime.utcfromtimestamp(time[i].astype('O')/1e9).month == datetime.utcfromtimestamp(time[i + cnt].astype('O')/1e9).month):
                        cnt += 1

                    if cnt >= 3:
                        var_new[i:i + cnt, j] = 1
                    i += cnt
                else:
                    i += 1

        ds_new = xr.Dataset({
            'index': (('time', 'lon'), var_new),
            'time': ('time', time),
            'lon': ('lon', lon)
        })

        ofile_path = f"{odir}{ofname}_index_TM.nc"
        if os.path.exists(ofile_path):
            os.remove(ofile_path)

        ds_new.to_netcdf(ofile_path)
        ds.close()