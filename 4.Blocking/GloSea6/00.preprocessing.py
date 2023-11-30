import os
import subprocess
import glob

def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")

model = "GloSea6"
wdir = "/data03/Glosea5/jhsim/NCL2PYTHON/METRIC_NCL"
ifile = f"{wdir}/DATA/{model}/ORIGINAL/"
ofile = f"/data03/Glosea5/jhsim/NCL2PYTHON/METRIC_PYTHON/4.Blocking/{model}/ifile/"

# 디렉토리 생성
os.makedirs(ofile, exist_ok=True)

for yy in range(1993, 2017):
    yyp1 = yy + 1

    for mm in ["02", "10", "12"]:
        for tvar in "hgt":
            ivar="hgt_e"
            fvar="hgt500"
            count=1
            for ens in list(range(1,8)):
                print(ens)
                iens=f'{count:03d}'
                nens=f'{count:02d}'
                
                input_file = f"{ifile}/{ivar}/{fvar}_{yy}{mm}01_{iens}.nc"
                output_file = f"{ofile}/{model}.{yy}{mm}01_{nens}.nc"
                
                if mm == "10":
                    time=61
                elif mm == "12":
                    time=62
                elif mm == "02":
                    time=59
                    input_file = f"{ifile}/{ivar}/{fvar}_{yyp1}{mm}01_{iens}.nc"
                    output_file = f"{ofile}/{model}.{yyp1}{mm}01_{nens}.nc"
                    
                print(f"{input_file} \n--> {output_file}")

                tmp_file = f"{ofile}/{model}.{yy}{mm}01_{nens}_tmp.nc"
                tmp2_file = f"{ofile}/{model}.{yy}{mm}01_{nens}_tmp2.nc"

                commands = [
                        f"cdo seltimestep,1/{time} {input_file} {tmp_file}",
                        f"cdo remapbil,r288x145 {tmp_file} {output_file}"
                    ]
                for cmd in commands:
                    run_command(cmd)

                # 임시 파일 제거
                flist = glob.glob(f"{ofile}/{model}*_tmp*.nc")
                for fname in flist:
                    os.remove(fname)
                count += 1
