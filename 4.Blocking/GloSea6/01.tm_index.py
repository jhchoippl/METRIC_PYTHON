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
root = f"{wdir}/4.Blocking/{model}/"
ofile = f"/data03/Glosea5/jhsim/NCL2PYTHON/METRIC_PYTHON/4.Blocking/{model}"

os.chdir(root)

idir = f"{ofile}/ifile"
ifname = f"{model}."
odir = f"{ofile}/ofile"
ofname0 = f"{ifname}IB.index."
ofname1 = f"{ifname}IB.strength."

seasons = {
    "ON": ("1993", "2015", "1001"),
    "DJ": ("1993", "2015", "1201"),
    "FM": ("1994", "2016", "0201")
}

nlat = 145
nlon = 288
nlon2 = 288

# User Set
os.makedirs(odir, exist_ok=True)

for nmm, (syr, eyr, season) in seasons.items():
    for ny in range(int(syr), int(eyr) + 1):
        for enum in range(1, 8):
            enum_str = f"{enum:02d}"
            ifile = f"{idir}/{ifname}{ny}{season}_{enum_str}.nc"

            latN = 80
            lat0 = 60
            latS = 40
            critN = "ltc,-10"
            critS = "gtc,0"
            delta = [-5, 0, 5]
            DLAT = 1

            ofile1 = f"{odir}/{ofname0}{ny}{season}_{enum_str}.nc"
            ofile2 = f"{odir}/{ofname1}{ny}{season}_{enum_str}.nc"

            for i, d in enumerate(delta):
                N = latN + d
                O = lat0 + d
                S = latS + d

                commands = [
                    f"cdo sellonlatbox,0,360,{N - DLAT},{N + DLAT} {ifile} tmp_N0",
                    f"cdo mermean tmp_N0 tmp_N",
                    f"cdo sellonlatbox,0,360,{O - DLAT},{O + DLAT} {ifile} tmp_O0",
                    f"cdo mermean tmp_O0 tmp_O",
                    f"cdo sellonlatbox,0,360,{S - DLAT},{S + DLAT} {ifile} tmp_S0",
                    f"cdo mermean tmp_S0 tmp_S",
                    f"cdo sub tmp_N tmp_O tmp",
                    f"cdo divc,{N - O} tmp tmp_GN",
                    f"cdo sub tmp_O tmp_S tmp",
                    f"cdo divc,{O - S} tmp tmp_GS",
                    f"cdo {critN} tmp_GN tmp_BN",
                    f"cdo {critS} tmp_GS tmp_BS",
                    f"cdo ensmin tmp_BN tmp_BS {ofile1}.delta{i}",
                    f"cdo ifthen {ofile1}.delta{i} tmp_GS {ofile2}.delta{i}"
                ]

                for cmd in commands:
                    run_command(cmd)

                run_command(f"rm -f tmp*")

            commands = [
                f"cdo ensmax {ofile1}.delta* {ofile1}",
                f"cdo ensmax {ofile2}.delta* {ofile2}",
                f"rm -f {ofile1}.delta* {ofile2}.delta*"
            ]

            for cmd in commands:
                run_command(cmd)
                
            flist = glob.glob(f"{ofile1}.delta* {ofile2}.delta*")
            for fname in flist:
                os.remove(fname)