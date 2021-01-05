# Trend plots of Ocean Color for the IPCC report
These scripts download, process, and plot trend data of the version 4.2 OC-CCI L4 satellite derived Chlorophyll product.

## Installation
All code is written in python 3.7+ and mainly uses commonly available packages. The plotting functions are based on cartopy which can be a bit difficult to install, it therefore advisabel to use the [conda](https://docs.conda.io/en/latest/miniconda.html]) package manager:

```
conda env create -f environment.yml
conda activate ipcc
```

## Run the code

Download, load, and process the OC-CCI files as follows:

```
>>> import chl_analysis
>>> chl_analysis.process()
```

Generate the figures with:

```
>>> chl_analysis.plot_clim()
>>> chl_analysis.plot_chl_trend()
```
