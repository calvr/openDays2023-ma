#!/usr/bin/env python3
# vim: set fileencoding=utf-8 fileformat=unix :
# -*- coding: utf-8 -*-
# vim: set ts=8 et sw=4 sts=4 sta :

import os
import urllib.request
import numpy as np
import xarray as xr
import scipy as sp
import scipy.stats
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
warnings.filterwarnings( "ignore", module = "matplotlib*" )


url = 'https://github.com/calvr/openDays2023-ma/raw/main/capel_trimmed.nc'
dst = os.path.join(os.getcwd(), os.path.basename(url))
if not os.path.isfile(dst):
    urllib.request.urlretrieve(url, dst)

assert os.path.isfile(dst), f"Failed to access dataset! Something went wrong with download, please repeat..."
print(f"{os.path.basename(dst)} copied to {os.path.dirname(dst)}/")


#######
##
## Define inputs
##
#######

ncFileName = dst

# V80 Power Curve (public, https://topfarm.pages.windenergy.dtu.dk/PyWake/notebooks/WindTurbines.html)
wsV80 = np.arange(0, 36, 1)
powerV80 = np.array([   0,    0,    0,    0,   66,  154,  282,  460,  696,  996, 1341,
                     1661, 1866, 1958, 1988, 1997, 1999, 2000, 2000, 2000, 2000, 2000,
                     2000, 2000, 2000, 2000,    0,    0,    0,    0,    0,    0,    0,
                        0,    0,    0])
ctV80 = np.array([0.   , 0.   , 0.   , 0.   , 0.818, 0.806, 0.804, 0.805, 0.806,
                  0.807, 0.793, 0.739, 0.709, 0.409, 0.314, 0.249, 0.202, 0.167,
                  0.14 , 0.119, 0.102, 0.088, 0.077, 0.067, 0.06 , 0.053, 0.   ,
                  0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.])
rhoV80 = 1.225
diameterV80 = 80


## define WS bins
wsBinLimits = np.arange(0, 36, 1)
wsBins = .5 * (wsBinLimits[1:] + wsBinLimits[:-1])
nWsBins = len(wsBins)


## define WD bins
make_int_if_whole_number = lambda x: int(x) if x.is_integer() else x
nWdBins = 12
wdBinSize = make_int_if_whole_number(360 / nWdBins)
wdHalfBinSize = make_int_if_whole_number(wdBinSize / 2)
wdBins = np.arange(0, 360, wdBinSize)
wdBinLimits = np.append(wdBins, 360) - wdHalfBinSize



#######
##
## Read data
## We want a Pandas DataFrame (suitable for time-series & tabular data) 
##
#######

## method 1: use netCDF4 directly
# import netCDF4
# nc = netCDF4.Dataset(ncFileName, 'r')
# signals = list(nc.variables)  # nc.variables.keys()
# times = nc.variables['time']  # get variable with times
# times = netCDF4.num2date(times[:], times.units)  # convert to Julian day
# times = pd.Series(times, name='time')  # get Pandas Series from array
# df = pd.DataFrame({k: nc.variables[k][:] for k in signals[1:]}, index=times)

## method 2: simply use XArray (faster due to low-level routines)
import xarray as xr
ds = xr.open_dataset(ncFileName)
df = ds.to_dataframe()


#######
##
## Scrutinize data
##
#######

print(df.columns)

wsSignalName = 'ws50_m1'  # wind speed signal name
wdSignalName = 'wd40_m1'  # wind direction signal name

print(f'checking metadata for columns {wsSignalName} and {wdSignalName}...')
print(ds[wsSignalName])
print(ds[wdSignalName])

# ## Separate per met-mast
# m1Signals = df.columns[df.columns.str.endswith('_m1')]
# m2Signals = df.columns[df.columns.str.endswith('_m2')]
# m3Signals = df.columns[df.columns.str.endswith('_m3')]
# 
# ## Separate mast 1 per WS and WD signals
# wsM1Signals = m1Signals[m1Signals.str.startswith('ws')]
# wdM1Signals = m1Signals[m1Signals.str.startswith('wd')]
# 
# ## Get highest WS and WD signals
# ws = df[wsM1Signals[0]]
# wd = df[wdM1Signals[0]]

ws = df[wsSignalName]
wd = df[wdSignalName]

invalid = ws.isna() | wd.isna()  # start by those already marked as NaN
invalid = invalid | (wd >= 360) | (wd < 0) # disregard out-of-bounds wd
invalid = invalid | ~(ws > 0) | (ws > 99) # disregard ws <= 0 and unreasonable high values

ws[invalid] = np.nan
wd[invalid] = np.nan

availability = sum(~ws.isna()) / ws.size
print(f"availability = {availability*100:.1f}%")

timeRange = ws.index.max() - ws.index.min()
nExpectedRecords = int(timeRange.total_seconds() / 600)
availability = sum(~ws.isna()) / nExpectedRecords
print(f"""
if we subtract the timestamps we get
the following time range: {timeRange}
which is equivalent to {nExpectedRecords} periods of 10 minutes

so the correct availability is:
availability = {availability*100:.1f}%
""")

#ws = ws.dropna()
#wd = wd.dropna()


#######
##
## Fit Weibull distribution to whole data
##
#######

def fitWeibull_mle(u):
    """
    Fit a Weibull distribution to a sample of WindSpeed values through a
    generic Maximum Likelihood Estimation (MLE) approach.
    """
    u = np.atleast_1d(np.copy(u))
    if 0 == len(u) or np.isnan(u).all() or not (np.nanmax(u) > 0):
        return np.nan, 2.0
    K, _, A = sp.stats.weibull_min.fit(u[~np.isnan(u)], 2, floc=0, scale=np.nanmean(u))
    A = A if K > 0 and A > 0 else np.nan
    return A, K

Afull, Kfull = fitWeibull_mle(ws)

CDF = lambda ws, A, K: 1 - np.exp(-(ws/A)**K)
iCDF = lambda P, A, K: A * (-np.log(1 - P))**(1./K)
PDF = lambda ws, A, K: K/A * (ws/A)**(K-1) * np.exp(-(ws/A)**K)


nValidRecords = sum(~ws.isna())  # we disregard missing values
#nValidRecords = nExpectedRecords  # we assume missing values distribution to be same of available record
wsHist, _ = np.histogram(ws, bins=wsBinLimits, density=False)
wsHistDensity = wsHist/nValidRecords/np.diff(wsBinLimits)


with plt.xkcd():
    #plt.hist(ws, bins=wsBinLimits, density=True, histtype='stepfilled', edgecolor='k', facecolor='lightgrey', label="histogram")
    #plt.bar(wsBins, wsHistDensity, width=1, align='center', facecolor='lightgrey', edgecolor='None', linewidth=0)
    plt.fill_between(wsBins, wsHistDensity, step="mid", facecolor='lightgrey', edgecolor='None', label="histogram")
    plt.step(wsBins, wsHistDensity, where='mid', c='k')
    wsBinsFine = np.arange(0, wsBinLimits.max(), .1)
    plt.plot(wsBinsFine, PDF(wsBinsFine, Afull, Kfull), c='xkcd:teal', ls='-', lw=3, label='Weibull fitting')
    plt.gca().set_xlim(left=0, right=wsBinLimits.max())
    plt.gca().set_ylim(bottom=0)
    plt.legend(frameon=False)
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Probability density")
    plt.tight_layout(pad=0.5, h_pad=None, w_pad=None, rect=None)
    plt.show()


#######
##
## Yield estimate
##
#######

with plt.xkcd():
    plt.step(wsBins, wsHistDensity, where='mid', c='k')
    plt.fill_between(wsBins, wsHistDensity, step="mid", facecolor='lightgrey', edgecolor='None', label="histogram")
    wsBinsFine = np.arange(0, wsBinLimits.max(), .1)
    plt.plot(wsBinsFine, PDF(wsBinsFine, Afull, Kfull), c='xkcd:teal', ls='-', lw=3, label='Weibull fitting')
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Probability density", color='xkcd:teal')
    plt.gca().set_xlim(left=0, right=wsBinLimits.max())
    plt.gca().set_ylim(bottom=0)
    axTwin = plt.gca().twinx()
    axTwin.plot(wsV80, powerV80, c='xkcd:orange', ls='-', lw=3, label='V80 power curve')
    axTwin.set_ylim(bottom=0)
    axTwin.set_ylabel('Power (kW)', color='xkcd:orange')
    #plt.legend(frameon=False)
    plt.tight_layout(pad=0.5, h_pad=None, w_pad=None, rect=None)
    plt.show()


## Based on CDF
wsV80edges = np.concatenate([[wsV80[0]], 0.5*(wsV80[1:] + wsV80[:-1]), [wsV80[-1]]])
wsProb = CDF(wsV80edges[1:], Afull, Kfull) - CDF(wsV80edges[:-1], Afull, Kfull)
yieldV80 = sum(wsProb * powerV80)
print(f"""
Values based on Weibull fitting (using the CDF)
Yield = {yieldV80:g} kWh
AEP   = {yieldV80 * 365.25 * 24 / 1000:g} MWh/year
CF    = {yieldV80/ max(powerV80) * 100:.1f}%
""")


## Based on PDF
#wsV80mid = 0.5*(wsV80[1:] + wsV80[:-1])
#wsV80delta = wsV80[1:] - wsV80[:-1]
#powerV80mid = 0.5*(powerV80[1:] + powerV80[:-1])
#yieldV80 = sum(PDF(wsV80mid, Afull, Kfull) * wsV80delta * powerV80mid)
#print(f"""
#Values based on Weibull fitting (using the PDF)
#Yield = {yieldV80:g} kWh
#AEP   = {yieldV80 * 365.25 * 24 / 1000:g} MWh/year
#CF    = {yieldV80/ max(powerV80) * 100:.1f}%
#""")


#powerV80mid = 0.5*(powerV80[1:] + powerV80[:-1])
#wsProbHist = wsHist/nValidRecords
#yieldV80 = sum(wsProbHist * powerV80mid)
#print(f"""
#Values based on the histogram
#Yield = {yieldV80:g} kWh
#AEP   = {yieldV80 * 365.25 * 24 / 1000:g} MWh/year
#CF    = {yieldV80/ max(powerV80) * 100:.1f}%
#""")


#######
##
## Analysis per sector
##
#######

neg_wd = lambda x, dx=-180: np.mod(x - dx, 360) + dx

print(f"""
What does the `neg_wd` function do?
neg_wd(  0, -15) = {neg_wd(  0, -15)}
neg_wd(  5, -15) = {neg_wd(  5, -15)}
neg_wd( 10, -15) = {neg_wd( 10, -15)}

neg_wd(343, -15) = {neg_wd(343, -15)}
neg_wd(344, -15) = {neg_wd(344, -15)}
neg_wd(345, -15) = {neg_wd(345, -15)}
neg_wd(346, -15) = {neg_wd(346, -15)}
neg_wd(347, -15) = {neg_wd(347, -15)}
""")

print(f"""
nWdBins     = {nWdBins}
wdBins      = {wdBins}
wdBinLimits = {wdBinLimits}
""")


wdSectors = pd.cut(neg_wd(wd, wdBinLimits[0]), wdBinLimits, right=False)
wsInSectors = ws.groupby(wdSectors)
#wdBinCount = wdSectors.groupby(wdSectors).count()
wdBinCount = wsInSectors.count()
print(f"WD bin counts:\n{wdBinCount}")

frequency = wdBinCount / float(wdBinCount.sum())
print(f"Frequency (%) per WD sector:\n{(frequency * 100)}")

## Plot wind rose
with plt.xkcd():
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.grid(False)
    ax.set_aspect(1.0)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.radians(wdBins))
    ax.plot(np.radians(np.append(wdBins, wdBins[0])), np.append(frequency, frequency[0]) * 100, label='frequency (%)', c='xkcd:teal', marker='.', ls='-')
    # ax.plot(np.append(a, a[0]), np.append(k, k[0]), label='Weibull k', c='xkcd:teal', marker='.', ls='-')
    # ax.plot(np.append(a, a[0]), np.append(A, A[0]), label='Weibull A (m/s)', c='xkcd:magenta', marker='.', ls='-')
    ax.legend(loc='lower right', bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(1, 0), frameon=False) 
    ax.xaxis.grid(color='k', linestyle='--', lw=.5, alpha=0.5)
    #ax.yaxis.grid(color='k', linestyle='--', lw=.5, alpha=0.5)
    plt.gcf().tight_layout(pad=0.5, h_pad=None, w_pad=None, rect=None)
    plt.show()




# AK = wsInSectors.apply(fitWeibull_mle)
# weibPerSector = pd.DataFrame(index=AK.index)
# weibPerSector['A'], weibPerSector['K'] = zip(*AK)
# print(weibPerSector)

weibPerSector = pd.DataFrame(index=wdBins, columns=['A', 'K'])
for i, k in zip(weibPerSector.index, wsInSectors.groups.keys()):
    weibPerSector.loc[i, ['A', 'K']] = fitWeibull_mle(wsInSectors.get_group(k))

print(weibPerSector)




## Plot wind rose
u25 = np.array([iCDF(.25, A, K) for A, K in zip(weibPerSector['A'], weibPerSector['K'])])
u50 = np.array([iCDF(.50, A, K) for A, K in zip(weibPerSector['A'], weibPerSector['K'])])
u75 = np.array([iCDF(.75, A, K) for A, K in zip(weibPerSector['A'], weibPerSector['K'])])
u95 = np.array([iCDF(.95, A, K) for A, K in zip(weibPerSector['A'], weibPerSector['K'])])
u99 = np.array([iCDF(.99, A, K) for A, K in zip(weibPerSector['A'], weibPerSector['K'])])
with plt.xkcd():
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.grid(False)
    ax.set_aspect(1.0)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    a = np.radians(np.append(wdBins, wdBins[0]))
    ax.set_xticks(a)
    width = 2*np.pi/float(len(a))
    cmap = mpl.cm.viridis
    umax = np.ceil(max(u99)/2.)*2
    c25 = np.array(u25 / umax * (cmap.N-1)).astype(int)
    c50 = np.array(u50 / umax * (cmap.N-1)).astype(int)
    c75 = np.array(u75 / umax * (cmap.N-1)).astype(int)
    c95 = np.array(u95 / umax * (cmap.N-1)).astype(int)
    c99 = np.array(u99 / umax * (cmap.N-1)).astype(int)
    p = np.array([0, .25, .50, .75, .95, .99, 1])  # we want to pass from CDF to PMF values
    dp = np.diff(p)  # mind that the last is not needed
    f = np.append(frequency, frequency[0]) * 100
    ax.bar(a, dp[0]*f, width, align='center', color=cmap(c25), zorder=10)
    ax.bar(a, dp[1]*f, width, bottom=dp[0]*f, align='center', color=cmap(c50), zorder=10)
    ax.bar(a, dp[2]*f, width, bottom=np.sum(dp[:2])*f, align='center', color=cmap(c75), zorder=10)
    ax.bar(a, dp[3]*f, width, bottom=np.sum(dp[:3])*f, align='center', color=cmap(c95), zorder=10)
    ax.bar(a, dp[4]*f, width, bottom=np.sum(dp[:4])*f, align='center', color=cmap(c99), zorder=10)
    #ax.set_title('Frequency of occurrence (%)', ha='left')
    ax.set_xlabel('Bins: [25%, 50%, 75%, 95%, 99%] percentiles', ha='left', fontsize='x-small')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=umax))
    sm.set_array([])
    cax = plt.colorbar(sm, pad=0.08)
    cax.ax.set_title('Wind speed (m/s)', fontsize='small', pad=10)
    ax.xaxis.grid(color='k', linestyle=':', lw=.5, alpha=0.3)
    ax.yaxis.grid(color='k', linestyle=':', lw=.5, alpha=0.3)
    plt.gcf().tight_layout(pad=0.5, h_pad=None, w_pad=None, rect=None)
    plt.show()



with plt.xkcd():
    from cycler import cycler
    myCycler = cycler(color=['teal', 'skyblue',
                             'steelblue', 'indigo', 'orchid',
                             'palevioletred', 'crimson', 'orangered',
                             'darkorange', 'darkgoldenrod', 'mediumseagreen',
                             'mediumaquamarine'])
    plt.gca().set_prop_cycle(myCycler)
    wsBinsFine = np.arange(0, wsBinLimits.max(), .1)
    for i in weibPerSector.index:
        plt.plot(wsBinsFine, PDF(wsBinsFine, weibPerSector.loc[i,'A'], weibPerSector.loc[i,'K']), ls='-', lw=2, label=f'{i}')
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Probability density", color='xkcd:teal')
    plt.gca().set_xlim(left=0, right=wsBinLimits.max())
    plt.gca().set_ylim(bottom=0)
    plt.legend(frameon=False, fontsize='small')
    axTwin = plt.gca().twinx()
    axTwin.plot(wsV80, powerV80, c='xkcd:orange', ls='-', lw=3, label='V80 power curve')
    axTwin.set_ylim(bottom=0)
    axTwin.set_ylabel('Power (kW)', color='xkcd:orange')
    #plt.legend(frameon=False)
    plt.tight_layout(pad=0.5, h_pad=None, w_pad=None, rect=None)
    plt.show()



wsV80edges = np.concatenate([[wsV80[0]], 0.5*(wsV80[1:] + wsV80[:-1]), [wsV80[-1]]])
for i, k in zip(weibPerSector.index, wsInSectors.groups.keys()):
    weibPerSector.loc[i, ['A', 'K']] = fitWeibull_mle(wsInSectors.get_group(k))
    wsProb = (   CDF(wsV80edges[1:], weibPerSector.loc[i,'A'], weibPerSector.loc[i,'K'])
               - CDF(wsV80edges[:-1], weibPerSector.loc[i,'A'], weibPerSector.loc[i,'K'])
             )
    weibPerSector.loc[i,'yield'] = sum(wsProb * powerV80)


print(weibPerSector)


## Plot yield rose
with plt.xkcd():
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.grid(False)
    ax.set_aspect(1.0)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.radians(wdBins))
    ax.plot(np.radians(np.append(wdBins, wdBins[0])), np.append(weibPerSector['yield'], weibPerSector['yield'][0]), label='Yield (kWh)', c='xkcd:orange', marker='.', ls='-')
    # ax.plot(np.append(a, a[0]), np.append(k, k[0]), label='Weibull k', c='xkcd:teal', marker='.', ls='-')
    # ax.plot(np.append(a, a[0]), np.append(A, A[0]), label='Weibull A (m/s)', c='xkcd:magenta', marker='.', ls='-')
    ax.legend(loc='lower right', bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(1, 0), frameon=False)
    ax.xaxis.grid(color='k', linestyle='--', lw=.5, alpha=0.5)
    #ax.yaxis.grid(color='k', linestyle='--', lw=.5, alpha=0.5)
    plt.gcf().tight_layout(pad=0.5, h_pad=None, w_pad=None, rect=None)
    plt.show()




