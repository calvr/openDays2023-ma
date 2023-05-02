# openDays2023-ma

Notebook and data for OpenDays2023 workshop - M&amp;A

## Workshops


### Workshop 1: Estimate Wind Turbine production under a given climate

The objectives of this workshop are:

* Demonstrate how to read and process a meteorological time series so to get a climate.
* Compute a wind-turbine yield by applying its power curve to the climate.
* Expand the analysis to consider sector-wise variations of the climate.

Run the tutorial via a free cloud platform: <a href="https://colab.research.google.com/github/calvr/openDays2023-ma/blob/main/openDays_workshop1.ipynb"> <img src = "https://colab.research.google.com/assets/colab-badge.svg" alt = "Colab">

### Workshop 2: Calculate Wind Turbine wakes interaction using pywake

Objectives:

* Demonstrate an open-source python package - PyWake (https://topfarm.pages.windenergy.dtu.dk/PyWake/index.html) and its applicability via a simplified case. 
* Understand how to model a wind farm interaction given a climate and a wind turbine model.
* Show the importance of taking the wake effects into consideration when modelling/designing a wind turbine park.

Run the tutorial via a free cloud platform: <a href="https://colab.research.google.com/github/calvr/openDays2023-ma/blob/main/openDays_workshop2.ipynb"> <img src = "https://colab.research.google.com/assets/colab-badge.svg" alt = "Colab">

### Workshop 3: Optimize the layout of a wind farm park

Objectives:

* Demonstrate an open-source python package for wind turbine parks optimization - Topfarm (https://topfarm.pages.windenergy.dtu.dk/TopFarm2/index.html).
* Show a gradient-based optimization method and visually analyse how it behaves.
* Optimize the layout modelled in the Workshop 2 to understand the convenience of using optimization.

Run the tutorial via a free cloud platform: <a href="https://colab.research.google.com/github/calvr/openDays2023-ma/blob/main/openDays_workshop3.ipynb"> <img src = "https://colab.research.google.com/assets/colab-badge.svg" alt = "Colab">


## Acknoledgements on datasets

The NetCDF file [`capel_trimmed.nc`](https://github.com/calvr/openDays2023-ma/blob/main/capel_trimmed.nc) is a reduced version of the `capel_all.nc` data available from [DOI 10.11583/DTU.14135627](https://doi.org/10.11583/DTU.14135627) whose site details are referred in [Database on Wind Characteristics for Capel Cynon resource data](https://gitlab.windenergy.dtu.dk/fair-data/winddata-revamp/winddata-documentation/-/blob/master/capel.md). This dataset was made available by [DTU Research Platform](https://data.dtu.dk/) under a [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license. The author acknoledges both [DTU Data](https://data.dtu.dk/) and [Future Energy Solutions, AEA Technology](http://www.aeat.co.uk/cms/) for making this data available to the public domain.
