import os
import subprocess
import glob
from cdo import Cdo
cdo = Cdo()

# 경로 설정
root = "/data03/Glosea5/jhsim/NCL2PYTHON/METRIC_NCL/"
ipath = os.path.join(root, "DATA/GloSea5/ORIGINAL/")
opath = '/data03/Glosea5/jhsim/NCL2PYTHON/METRIC_PYTHON/DATA/GloSea5/POST/'

# 디렉토리 생성
os.makedirs(opath, exist_ok=True)

zaxis_content = """zaxistype = pressure
size      = 9
name      = level
longname  = pressure
units     = hPa
levels    = 1000 925 850 500 300 200 100 50 10"""

with open(os.path.join(opath, "myzaxis_all"), "w") as f:
    f.write(zaxis_content)
    
for var in ["t2m", "geo", "sic"]:  # 원본 스크립트에서는 ["t2m", "geo", "sic"] 사용됨
    # 변수 설정
    print(var)
    if var == "geo":
        ivar, fvar, ovar = "geo", "gh", "hgt"
    elif var == "t2m":
        ivar, fvar, ovar = "t2mp", "t", "t15m"
    elif var == "sic":
        ivar, fvar, ovar = "sice", "iceconc", "sic"

    # 디렉토리 생성
    os.makedirs(os.path.join(opath, ovar), exist_ok=True)

    # 연도 및 초기화 시간 루프
    for yy in range(1993, 2017):
        print(yy)
        if yy == 2016:
            inits=["0201"]
        else:
            inits=["1001", "1201", "0201"]
        for init in inits:
            imm = init[:2]

            # 파일 처리
            infile_pattern = f"{ivar}.{yy}{init}_????.nc"
            infile_path = os.path.join(ipath, var, infile_pattern)
            outfile_tmp2 = os.path.join(opath, ovar, f"{ovar}.{yy}{init}_ensmean_tmp2.nc")
            outfile_tmp1 = os.path.join(opath, ovar, f"{ovar}.{yy}{init}_ensmean_tmp1.nc")

            if var == "t2m":
                cdo.ensmean(input=infile_path, output=outfile_tmp2)
                cdo.seltimestep('1/100', input=outfile_tmp2, output=outfile_tmp1)
            elif var == "sic":
                infile_sic = os.path.join(ipath, ivar, f"{ivar}.{yy}{init}_ensmean.nc")
                subprocess.run(["cp", infile_sic, outfile_tmp1])
            else:
                cdo.ensmean(input=infile_path, output=outfile_tmp2)
                outfile_tmp3 = os.path.join(opath, ovar, f"{ovar}.{yy}{init}_ensmean_tmp3.nc")
                cdo.seltimestep('1/100', input=outfile_tmp2, output=outfile_tmp3)
                outfile_tmp4 = os.path.join(opath, ovar, f"{ovar}.{yy}{init}_ensmean_tmp4.nc")
                cdo.sellevel('100000,92500,85000,50000,30000,20000,10000,5000,1000', input=outfile_tmp3, output=outfile_tmp4)
                cdo.setzaxis(os.path.join(opath, "myzaxis_all"), input=outfile_tmp4, output=outfile_tmp1)

                # 임시 파일 삭제
                flist = glob.glob(f"{opath}/{ovar}/*_tmp[2-5].nc")
                for fname in flist:
                    os.remove(fname)

            # 변수 이름 변경
            outfile_tmp2 = os.path.join(opath, ovar, f"{ovar}.{yy}{init}_ensmean_tmp2.nc")
            cdo.chname(f"{fvar},{ovar}", input=outfile_tmp1, output=outfile_tmp2)

            # 재투영
            outfile_tmp3 = os.path.join(opath, ovar, f"{ovar}.{yy}{init}_ensmean_tmp3.nc")
            if var == "sic":
                cdo.remapbil("r360x180", input=outfile_tmp2, output=outfile_tmp3)
            else:
                cdo.remapbil("r288x145", input=outfile_tmp2, output=outfile_tmp3)

            # 순서 변경 및 이름 변경
            final_output = os.path.join(opath, ovar, f"{ovar}.{yy}{init}_ensmean.nc")
            subprocess.run(["ncpdq", "-a", "-level", outfile_tmp3, final_output])
            subprocess.run(["ncrename", "-d", "lon,longitude", "-v", "lon,longitude", final_output])
            subprocess.run(["ncrename", "-d", "lat,latitude", "-v", "lat,latitude", final_output])

            # 임시 파일 삭제
            flist = glob.glob(f"{opath}/{ovar}/*_tmp*.nc")
            for fname in flist:
                os.remove(fname)
