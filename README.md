# Trend plots of Ocean Color for the IPCC report
Code to generate trend plots of Chl concentrations based on the v4.2 OC-CCI L4 satellite derived Chlorophyll product. Downloading and processing the data require about 70GB of disk space and preferably a machine with 32GB RAM. The code is parallelized and will utilize multicore architectures. 

## Installation
All code is written in python 3.7+ and mainly uses commonly available packages. The plotting functions are based on cartopy which can be a bit difficult to install, it therefore advisable to use the [conda](https://docs.conda.io/en/latest/miniconda.html) package manager:

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
The parallelized setup is based on a machine with at least 32GB RAM. Change the *islice* attribute in the *process* function to a lower value if memory is an issue. For example:

```
>>> chl_analysis.process(islice=270)
```

Generate the figures with:

```
>>> chl_analysis.plot_chl_clim()
>>> chl_analysis.plot_hatched_chl_trend()
```
