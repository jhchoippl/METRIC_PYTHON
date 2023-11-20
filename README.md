# METRIC_PYTHON

## 분업
### 전처리 변환 및 코드 완성
ORG DIR=/data03/Glosea5/jhsim/NCL2PYTHON/METRIC_NCL
* 1.ORG DIR/1.ACC_RMSE/002_ACC_RMSE_SIC.ncl (HYLEE)
* 2.ORG DIR/2.AO/001_AO.ncl, Cal_AO_JRA55.ncl, Cal_AO_JRA55_PATTERN.ncl, Cal_AO_Model.ncl
* 3.ORG DIR/3.ART/001_ART.ncl, Cal_ART.ncl
* 4.ORG DIR/4.Blocking/003_Blocking_yearly.ncl, Cal_Frequency_ENS.ncl, Cal_Frequency.ncl

## 작업 진행 방법
### conda env 구성
```
conda create -n <envname> -c=conda-forge xarray cartopy pandas matplotlib cmaps geocat-viz cftime h5netcdf ipython ipykernel

conda activate <envname>
```
* 1.ACC_RMSE 디렉토리에 전처리 코드가 완성된 001_ACC_RMSE_T2M_GPH_plot.py 파일을 참고
* src 디렉토리의 파일을 참고
* 각 디렉토리의 data 디렉토리에 전처리가 완료된 파일을 참고

## Tip
* 전처리 부분은 수정을 할 필요가 없지만 시간이 오래걸리기 때문에 가급적 ipynb파일에서 2번 cell을 실행하고 3번 cell만 수정, 실행을 진행 권장
* 잘 모르겠다 싶으면 물어보기
* GPT 잘 활용하기