import os
import subprocess
import glob

def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")

model = "JRA55"
wdir = "/data03/Glosea5/jhsim/NCL2PYTHON/METRIC_NCL"
ifile = f"{wdir}/DATA/{model}/"
ofile = f"/data03/Glosea5/jhsim/NCL2PYTHON/METRIC_PYTHON/4.Blocking/{model}/ifile/"

# 디렉토리 생성
os.makedirs(ofile, exist_ok=True)

for yy in range(1993, 2017):
    yyp1 = yy + 1

    for mm in ["02", "10", "12"]:
        for tvar in ["hgt"]:
            count = 1
            nens = f"{count:02d}"

            # geo 처리

            if mm == "10":
                commands = [
                    f'cdo sellevel,500 {ifile}/{tvar}/{tvar}.day.{yy}10.nc {ofile}/JRA55.{yy}10_tmp.nc',
                    f'cdo sellevel,500 {ifile}/{tvar}/{tvar}.day.{yy}11.nc {ofile}/JRA55.{yy}11_tmp.nc',
                    f'cdo mergetime {ofile}/JRA55.{yy}10_tmp.nc {ofile}/JRA55.{yy}11_tmp.nc {ofile}/JRA55.{yy}1001_tmp.nc',
                    f'cdo seltimestep,1/61 {ofile}/JRA55.{yy}{mm}01_tmp.nc {ofile}/JRA55.{yy}{mm}01.nc'
                ]

            elif mm == "12":
                commands = [
                    f'cdo sellevel,500 {ifile}/{tvar}/{tvar}.day.{yy}12.nc {ofile}/JRA55.{yy}12_tmp.nc',
                    f'cdo sellevel,500 {ifile}/{tvar}/{tvar}.day.{yyp1}01.nc {ofile}/JRA55.{yyp1}01_tmp.nc',
                    f'cdo mergetime {ofile}/JRA55.{yy}12_tmp.nc {ofile}/JRA55.{yyp1}01_tmp.nc {ofile}/JRA55.{yy}1201_tmp.nc',
                    f'cdo seltimestep,1/62 {ofile}/JRA55.{yy}{mm}01_tmp.nc {ofile}/JRA55.{yy}{mm}01.nc'
                ]

            elif mm == "02":
                commands = [
                    f'cdo sellevel,500 {ifile}/{tvar}/{tvar}.day.{yyp1}02.nc {ofile}/JRA55.{yyp1}02_tmp.nc',
                    f'cdo sellevel,500 {ifile}/{tvar}/{tvar}.day.{yyp1}03.nc {ofile}/JRA55.{yyp1}03_tmp.nc',
                    f'cdo mergetime {ofile}/JRA55.{yyp1}02_tmp.nc {ofile}/JRA55.{yyp1}03_tmp.nc {ofile}/JRA55.{yyp1}0201_tmp.nc',
                    f'cdo seltimestep,1/59 {ofile}/JRA55.{yyp1}{mm}01_tmp.nc {ofile}/JRA55.{yyp1}{mm}01.nc'
                ]
                
            for cmd in commands:
                run_command(cmd)

            # 임시 파일 제거
            flist = glob.glob(f"{ofile}/{model}*_tmp*.nc")
            for fname in flist:
                os.remove(fname)
            count += 1
