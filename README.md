# Trend plots of Ocean Color for the IPCC report
These scripts download, process, and plot trend data of the version 4.2 OC-CCI L4 satellite derived Chlorophyll product.

## Installation
All code is written in python 3.7+ and mainly uses commonly available packages. It is probably a good idea to run the package in a virtual environment. Install prerequsite packages using

```
pip install -r requirements.txt
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
