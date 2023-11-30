# METRIC_PYTHON

## Setting Up the Conda Environment
```
# Create a new environment with the specified packages
conda create -n <envname> -c conda-forge xarray cartopy pandas matplotlib cmaps geocat-viz cftime ipython ipykernel statsmodels eofs python-cdo

# Activate the environment
conda activate <envname>

```
## Description of Files
* 0\. POST
    * Files for post-processing model data.
* 2\. AO
    * [001_AO.py](https://github.com/jhchoippl/METRIC_PYTHON/blob/main/2.AO/001_AO.py) for preprocessing and plotting.
    * [001_AO_plot.py](https://github.com/jhchoippl/METRIC_PYTHON/blob/main/2.AO/001_AO_plot.py) only for plotting preprocessed data.
    * [Cal_AO.py](https://github.com/jhchoippl/METRIC_PYTHON/blob/main/2.AO/Cal_AO.py) contains functions needed for AO preprocessing.
* 3\. ART
    * [001_ART.py](https://github.com/jhchoippl/METRIC_PYTHON/blob/main/3.ART/001_ART.py) for preprocessing and plotting.
    * [001_ART_plot.py](https://github.com/jhchoippl/METRIC_PYTHON/blob/main/3.ART/001_ART_plot.py) only for plotting preprocessed data.
    * [Cal_ART.py](https://github.com/jhchoippl/METRIC_PYTHON/blob/main/3.ART/Cal_ART.py) contains functions needed for ART preprocessing.
* 4\. Blocking
    * [003_Blocking_yearly.py](https://github.com/jhchoippl/METRIC_PYTHON/blob/main/4.Blocking/003_Blocking_yearly.py) for preprocessing and plotting.
    * [003_Blocking_yearly_plot.py](https://github.com/jhchoippl/METRIC_PYTHON/blob/main/4.Blocking/003_Blocking_yearly_plot.py) only for plotting preprocessed data.
    * Directory names matching model names contain files for data preprocessing.
* src
    * [READ_FILE.py](https://github.com/jhchoippl/METRIC_PYTHON/blob/main/src/READ_FILE.py) for reading post-processed data.
    * [NCL_FUNC.py](https://github.com/jhchoippl/METRIC_PYTHON/blob/main/src/NCL_FUNC.py) includes functions to implement NCL features in Python.