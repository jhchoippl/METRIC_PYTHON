import os
import subprocess
import glob
from cdo import Cdo

cdo = Cdo()

# 기본 경로 설정
root = "/data03/Glosea5/jhsim/NCL2PYTHON/METRIC_NCL/"
ipath = os.path.join(root, "DATA/GloSea6/ORIGINAL/")
opath = '/data03/Glosea5/jhsim/NCL2PYTHON/METRIC_PYTHON/DATA/GloSea6/POST/'

# 디렉토리 생성
os.makedirs(opath, exist_ok=True)

# myzaxis_all 파일 생성
zaxis_all_content = """zaxistype = pressure
size      = 9
name      = level
longname  = pressure
units     = hPa
levels    = 10 50 100 200 300 500 850 925 1000"""

with open(os.path.join(opath, "myzaxis_all"), "w") as f:
    f.write(zaxis_all_content)

# 각 압력 레벨별 myzaxis 파일 생성
for lev in [1000, 925, 850, 500, 300, 200, 100, 50, 10]:
    zaxis_content = f"""zaxistype = pressure
size      = 1
name      = level
longname  = pressure
units     = hPa
levels    = {lev}"""

    with open(os.path.join(opath, f"myzaxis_{lev}"), "w") as f:
        f.write(zaxis_content)

# 변수 처리
for var in ["sic", "t15m", "hgt"]:  # ["temp" "t15m" "hgt" "uwind" "vwind"]
    # 변수 설정
    if var == "hgt":
        ivar = fvar = ovar = "hgt"
    elif var == "t15m":
        ivar = fvar = ovar = "t15m"
    elif var == "sic":
        ivar = fvar = ovar = "sic"

    # 디렉토리 생성
    os.makedirs(os.path.join(opath, ovar), exist_ok=True)

    # 연도 및 초기화 시간 루프
    for yy in range(1993, 2017):
        print(yy)
        for init in ["1001", "1201", "0201"]:
            imm = init[:2]

            if var in ["t15m", "sic"]:
                infile = os.path.join(ipath, var, f"{ivar}_{yy}{init}_ensm.nc")
                outfile_tmp1 = os.path.join(opath, ovar, f"{ovar}.{yy}{init}_ensmean_tmp1.nc")
                cdo.seltimestep('1/100', input=infile, output=outfile_tmp1)
            else:
                # 레벨별 처리
                for lev in [1000, 925, 850, 500, 300, 200, 100, 50, 10]:
                    tlev = f"{lev:04d}"
                    infile = os.path.join(ipath, var, f"{ivar}{lev}_{yy}{init}_ensm.nc")
                    outfile_tmp2 = os.path.join(opath, ovar, f"{ovar}{lev}.{yy}{init}_ensmean_tmp2.nc")
                    outfile_tmp3 = os.path.join(opath, ovar, f"{ovar}{lev}.{yy}{init}_ensmean_tmp3.nc")
                    cdo.seltimestep('1/100', input=infile, output=outfile_tmp2)
                    cdo.setzaxis(os.path.join(opath, f"myzaxis_{lev}"), input=outfile_tmp2, output=outfile_tmp3)
                    cdo.chname(f"{fvar}{lev},{ovar}", input=outfile_tmp3, output=os.path.join(opath, ovar, f"{ovar}{tlev}.{yy}{init}_ensmean_tmp4.nc"))

                # 파일 병합
                merged_file = os.path.join(opath, ovar, f"{ovar}.{yy}{init}_ensmean_tmp5.nc")
                cdo.merge(input=f"{opath}/{ovar}/{ovar}*.{yy}{init}_ensmean_tmp4.nc", output=merged_file)
                cdo.setzaxis(os.path.join(opath, "myzaxis_all"), input=merged_file, output=outfile_tmp1)

                # 임시 파일 삭제
                flist = glob.glob(f"{opath}/{ovar}/*_tmp[2-5].nc")
                for fname in flist:
                    os.remove(fname)

            # 변수 이름 변경 및 재투영
            outfile_tmp2 = os.path.join(opath, ovar, f"{ovar}.{yy}{init}_ensmean_tmp2.nc")
            cdo.chname(f"{fvar},{ovar}", input=outfile_tmp1, output=outfile_tmp2)

            if var == "sic":
                target_grid = "r360x180"
            else:
                target_grid = "r288x145"

            outfile_tmp3 = os.path.join(opath, ovar, f"{ovar}.{yy}{init}_ensmean_tmp3.nc")
            cdo.remapbil(target_grid, input=outfile_tmp2, output=outfile_tmp3)

            # 순서 변경 및 이름 변경
            final_output = os.path.join(opath, ovar, f"{ovar}.{yy}{init}_ensmean.nc")
            subprocess.run(["ncpdq", "-a", "-level", outfile_tmp3, final_output])
            subprocess.run(["ncrename", "-d", "lon,longitude", "-v", "lon,longitude", final_output])
            subprocess.run(["ncrename", "-d", "lat,latitude", "-v", "lat,latitude", final_output])

            # 임시 파일 삭제
            flist = glob.glob(f"{opath}/{ovar}/*_tmp*.nc")
            for fname in flist:
                os.remove(fname)