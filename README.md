# METRIC_PYTHON

## 수정사항
### 추후 업데이트

## 작업 진행 방법
### conda env 구성
```
conda create -n <envname> -c=conda-forge xarray cartopy pandas matplotlib cmaps geocat-viz cftime h5netcdf ipython ipykernel

conda activate <envname>
```
 <!-- * exem(exemple) 디렉토리에 ipynb와 py 형식의 파일이 있음
 * ipynb 파일
	 * 1번째 cell : data 읽고 연평균 계산
	 * 2번째 cell : 전처리
	 * 3번째 cell : 그림 출력 코드
* py 파일
	* ipynb 파일이 읽기 어려운 경우 사용
	* 2번째, 3번째 cell을 병합한 코드

* ipynb 1번째 cell은 연평균을 계산한 nc파일을 따로 저장해뒀기 때문에 실행 할 필요가 없음 -->

## Tip
* 전처리 부분은 수정을 할 필요가 없지만 시간이 오래걸리기 때문에 가급적 ipynb파일에서 2번 cell을 실행하고 3번 cell만 수정, 실행을 진행 권장
