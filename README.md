# METRIC_PYTHON

## conda env 구성
```
conda create -n <envname> -c=conda-forge xarray cartopy pandas matplotlib cmaps geocat-viz cftime ipython ipykernel statsmodels eofs python-cdo

conda activate <envname>
```
## 파일별 설명
* 0\. POST
    * 모델 데이터 후처리를 위한 파일
* 2\. AO
    * [001_AO.py](https://github.com/jhchoippl/METRIC_PYTHON/blob/main/2.AO/001_AO.py)는 전처리 + plotting
    * [001_AO_plot.py](https://github.com/jhchoippl/METRIC_PYTHON/blob/main/2.AO/001_AO_plot.py)는 onpy plotting
    * [Cal_AO.py](https://github.com/jhchoippl/METRIC_PYTHON/blob/main/2.AO/Cal_AO.py)는 AO 전처리에 필요한 함수들
* 3\. ART
    * [001_ART.py](https://github.com/jhchoippl/METRIC_PYTHON/blob/main/3.ART/001_ART.py)는 전처리 + plotting
    * [001_ART_plot.py](https://github.com/jhchoippl/METRIC_PYTHON/blob/main/3.ART/001_ART_plot.py)는 onpy plotting
    * [Cal_ART.py](https://github.com/jhchoippl/METRIC_PYTHON/blob/main/3.ART/Cal_ART.py)는 AO 전처리에 필요한 함수들

* 4\. Blocking
    * [003_Blocking_yearly.py](https://github.com/jhchoippl/METRIC_PYTHON/blob/main/4.Blocking/003_Blocking_yearly.py)는 전처리 + plotting
    * [003_Blocking_yearly_plot.py](https://github.com/jhchoippl/METRIC_PYTHON/blob/main/4.Blocking/003_Blocking_yearly_plot.py)는 onpy plotting
    * 모델명 디렉토리는 데이터 전처리를 위한 파일
* src
    * [READ_FILE.py](https://github.com/jhchoippl/METRIC_PYTHON/blob/main/src/READ_FILE.py)은 후처리 된 데이터를 읽는 파일
    * [NCL_FUNC.py](https://github.com/jhchoippl/METRIC_PYTHON/blob/main/src/NCL_FUNC.py)은 NCL의 함수를 python에서 구현하기 위한 함수들