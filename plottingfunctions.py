from scipy import io
from scipy import optimize
from scipy import special as sc
from scipy import stats
from scipy.stats import t
import numpy as np
import pandas as pd 
import math
from numpy.random import default_rng
import numpy.random as random
import csv
import matplotlib.pyplot as plt
from collections import Counter
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)






def plot_example_lognormal(mu, variance):
    plt.rcParams["figure.figsize"] = (20,20)
    plt.rcParams["axes.labelsize"] = 25
    hfont = {'fontname':'Times New Roman'}
    plt.rcParams["xtick.labelsize"] = 24
    plt.rcParams["ytick.labelsize"] = 24
    plt.rcParams["font.family"] = "Times New Roman"
    mu = 3
    variance = 1
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

    x_small = x[np.exp(x)<=42]

    fig = plt.figure(1, (20,20))

    ax0 = fig.add_subplot(2,2,1)
    #plt.plot(x, stats.norm.pdf(x, mu, sigma), color=(0.5,0.5,1), linewidth=2)
    #plt.fill_between(x, 0, stats.norm.pdf(x, mu, sigma), color=(0.5,0.5,1))
    ax0.fill_between([xi for xi in x], 0, stats.norm.pdf(x, mu, sigma), color=(0.5,0.5,1))
    ax0.scatter((mu), stats.norm.pdf(mu, mu, sigma), label='Median$\mathregular{='+str(round((mu),2))+'}$')
    ax0.scatter((mu), stats.norm.pdf(mu, mu, sigma), label='Mean$\mathregular{='+str(round((mu),2))+'}$')
    #plt.xscale("log")
    ax0.set_xlabel("Lognormal distribution in logspace (equivalent to a Gaussian)", **hfont)
    ax0.legend(loc="lower right", fontsize=24)
    #plt.ylabel("Depth [m]", **hfont)
    ax1 = fig.add_subplot(2,2,2)
    ax1.fill_between([np.exp(xi) for xi in x], 0, stats.norm.pdf(x, mu, sigma), color=(0.5,0.5,1))
    ax1.scatter(np.exp(mu), stats.norm.pdf(mu, mu, sigma), label='Median$\mathregular{='+str(round(np.exp(mu),2))+'}$')
    ax1.scatter(np.exp(mu+0.5*(sigma**2)), stats.norm.pdf(mu+0.5*(sigma**2), mu, sigma), label='Mean$\mathregular{='+str(round(np.exp(mu+0.5*(sigma**2)),2))+'}$')
    ax1.set_xlabel("Lognormal distribution (exponential of Gaussian dist.)", **hfont)
    #ax1.legend(loc="lower right", fontsize=16)
    #plt.ylabel("Depth [m]", **hfont)


    ax2 = plt.axes([0,0,1,1])
    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(ax1, [0.4,0.4,0.5,0.5])
    ax2.set_axes_locator(ip)
    ax2.fill_between([np.exp(xi) for xi in x_small], 0, stats.norm.pdf(x_small, mu, sigma), color=(0.5,0.5,1))
    ax2.scatter(np.exp(mu), stats.norm.pdf(mu, mu, sigma), label='Median$\mathregular{='+str(round(np.exp(mu),2))+'}$')
    ax2.scatter(np.exp(mu+0.5*(sigma**2)), stats.norm.pdf(mu+0.5*(sigma**2), mu, sigma), label='Mean$\mathregular{='+str(round(np.exp(mu+0.5*(sigma**2)),2))+'}$')
    ax2.legend(loc="lower right", fontsize=24)
    # Mark the region corresponding to the inset axes on ax1 and draw lines
    # in grey linking the two axes.
    mark_inset(ax1, ax2, loc1=1, loc2=4, fc="none", ec='0.5')

    plt.savefig('_example_lognormal.pdf', format='pdf')
    plt.show()



"""DESCRIPTIVE/DATA SUMMARY PLOTTING FUNCTIONS"""

def plot_binned_histograms(binned_model, x_bounds=[50, 4000]):
    """
    given a set of bins, plot a histogram of each bin and overlay a gaussian distribution with the same mean and variance.
    also plot the deepest bin in realspace.
    """
    (sliced_flux, sliced_depth, sliced_mu, sliced_variance, sliced_mean_depth, power_law_params, boundaries) = binned_model
    num_bins = len(sliced_mu)
    ncols = 4
    nrows = 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(40, 20))

    plt.rcParams["axes.labelsize"] = 25
    hfont = {'fontname':'Times New Roman'}
    plt.rcParams["xtick.labelsize"] = 24
    plt.rcParams["ytick.labelsize"] = 24
    plt.rcParams["font.family"] = "Times New Roman"

    counter = 0
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i][j]

            # Plot when we have data
            if counter < len(sliced_flux):
                mu = sliced_mu[counter]
                sigma = np.sqrt(sliced_variance[counter])
                x = np.linspace(mu - 3.5*sigma, mu + 3.5*sigma, 500)
                d = np.log(sliced_flux[counter])
                ax.hist(d, bins=40, color='blue', alpha=0.5, density=True)
                depthrange=str(boundaries[counter]) + '-' + str(min(boundaries[counter+1], 4000)) + 'm'
                ax.plot(x, stats.norm.pdf(x, mu, sigma), color='black')
                ax.set_xlabel('Ln (POC flux [mg $\mathregular{ m^{-2} day^{-1}}$]), '+depthrange, **hfont, fontsize=30)
                ax.set_ylabel("Relative frequency", **hfont,fontsize=30)
            counter += 1
    
    
    
    # make one histogram in realspace to show that the distribution is only nice and gaussian in logspace
    ax = axes[1][3]
    ind = 6
    real_space_flux = sliced_flux[ind]
    n = len(real_space_flux)
    mu = sum(real_space_flux)/n
    sigma = np.sqrt(sum([(x - mu)**2 for x in real_space_flux])/n)
    x = np.linspace(mu - 3.5*sigma, mu + 3.5*sigma, 500)
    ax.hist(real_space_flux, bins=40, color='purple', alpha=0.5, density=True)
    depthrange=str(boundaries[ind]) + '-' + str(min(boundaries[ind+1], 4000)) + 'm'
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color='black')
    ax.set_xlabel('POC flux [mg $\mathregular{ m^{-2} day^{-1}}$], '+depthrange, **hfont, fontsize=30)
    ax.set_ylabel("Relative frequency", **hfont,fontsize=30)
    
    plt.savefig('_binned_histograms.pdf', format='pdf')
    plt.show()
    

# no longer using
def plot_binned_histograms_comparison(binned_model, alternate_binned_model):
    (sliced_flux, sliced_depth, sliced_mu, sliced_variance, sliced_mean_depth, power_law_params) = binned_model
    (sliced_flux2, sliced_depth2, sliced_mu2, sliced_variance2, sliced_mean_depth2, power_law_params2) = alternate_binned_model
    num_bins = len(sliced_mu)
    ncols = 3
    nrows = int(math.ceil(num_bins/3))
    height = (40/3)*nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(40, height))

    plt.rcParams["axes.labelsize"] = 25
    hfont = {'fontname':'Times New Roman'}
    plt.rcParams["xtick.labelsize"] = 24
    plt.rcParams["ytick.labelsize"] = 24
    plt.rcParams["font.family"] = "Times New Roman"
    # Lazy counter so we can remove unwanted axes
    counter = 0
    for i in range(nrows):
        for j in range(ncols):
            #print(j)
            ax = axes[i][j]

            # Plot when we have data
            if counter < len(sliced_flux):
                mu = sliced_mu[counter]
                sigma = np.sqrt(sliced_variance[counter])
                x = np.linspace(mu - 3.5*sigma, mu + 3.5*sigma, 500)
                d = np.log(sliced_flux[counter])
                d2 = np.log(sliced_flux2[counter])
                ax.hist(d, bins=40, color='blue', alpha=0.5, density=True)
                ax.hist(d2, bins=40, color='red', alpha=0.5, density=True)
                depthrange=str(round(min(sliced_depth[counter]), 2)) + '-' + str(round(max(sliced_depth[counter]), 2)) + 'm'
                #ax.axvspan(mu,mu, color='black', alpha=0.5)
                ax.plot(x, stats.norm.pdf(x, mu, sigma), color='black')
                ax.set_xlabel('Ln (POC flux [mg $\mathregular{ m^{-2} day^{-1}}$]), '+depthrange, **hfont, fontsize=30)
                ax.set_ylabel("Relative frequency", **hfont,fontsize=30)
                #leg = ax.legend(loc='upper right')
                #leg.draw_frame(False)


                

            counter += 1
    #plt.savefig('fig/1_skewness.pdf', format='pdf')
    ax.set_axis_off()
    plt.show()

    

def plot_world_scatterplot(sediment_trap_data_shallow, deep_data):
    """
    plot the location distribution of two kinds of data
    """
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    lat_1 = [x.astype(float) for x in np.array(sediment_trap_data_shallow)[:,3]]
    lon_1 = [x.astype(float) for x in np.array(sediment_trap_data_shallow)[:,4]]
    
    lat_2 = [x.astype(float) for x in np.array(deep_data)[:,3]]
    lon_2 = [x.astype(float) for x in np.array(deep_data)[:,4]]
    ax.scatter(lon_2, lat_2, color = 'blue', label = 'Observations at or below 1000m')
    ax.scatter(lon_1, lat_1, color = 'green', label = 'Observations above 1000m')
    plt.legend(loc=(0.045,0.03), fontsize=16)
    
    plt.savefig('_world_scatterplot.pdf', format='pdf')
    plt.show()
    


    
    
    
    
    
    
    
    
    
"""BINNED POWER LAW MODEL PLOTTING FUNCTIONS"""

def plot_binned_model_scatterplot(binned_model, x_bounds=[50, 4000], reference_depth = 100):
    """
    given parameters for a power law fit to a binned model:
    make a scatterplot of average bin depth vs average bin flux,
    add error bars equal to the standard deviation of the points in each bin from the logmean.
    """
    (sliced_flux, sliced_depth, sliced_mu, sliced_variance, sliced_mean_depth, power_law_params, boundaries) = binned_model
    sliced_st_dev = [np.sqrt(x) for x in sliced_variance]
    xd=np.linspace(x_bounds[0], x_bounds[1],100)
    A, B, var = power_law_params['A'], power_law_params['B'], power_law_params['variance']
    aprint = str(round(A, 1))
    bprint = str(round(B, 2))
    varprint = str(round(var, 2))
    plt.rcParams["figure.figsize"] = (20,20)
    plt.rcParams["axes.labelsize"] = 25
    hfont = {'fontname':'Times New Roman'}
    plt.rcParams["xtick.labelsize"] = 24
    plt.rcParams["ytick.labelsize"] = 24
    plt.rcParams["font.family"] = "Times New Roman"
    
    # plot
    plt.subplot(2, 2, 1)
    plt.yticks(fontname = "Times New Roman")
    plt.xticks(fontname = "Times New Roman")
    
    # mean power law
    plt.plot(A+B*np.log(xd/reference_depth), xd, color='black', linewidth=3, label = '$\mathregular{'+aprint+'\, '+bprint+'\, ln(z/'+str(reference_depth)+')}$')
    # binned mean vs mean depth
    plt.scatter(sliced_mu, sliced_mean_depth, color='red', linewidth=3, label='average logscale flux of binned data')
    plt.errorbar(sliced_mu, sliced_mean_depth, xerr = sliced_st_dev, capsize = 7, capthick=2, elinewidth=1.5, ls='none', color='red', linewidth=3)
    plt.gca().invert_yaxis()
    plt.xlabel("Binned average of Ln (POC flux [mg $\mathregular{ m^{-2} day^{-1}}$])", **hfont)
    plt.ylabel("Depth [m]", **hfont)
    plt.legend(loc="lower right", fontsize=16)
    
    # plot in logspace
    plt.subplot(2, 2, 2)
    plt.yticks(fontname = "Times New Roman")
    plt.xticks(fontname = "Times New Roman")
    
    # mean power law
    plt.plot(A+B*np.log(xd/reference_depth), np.log10(xd), color='black', linewidth=3, 
             label = '$\mathregular{'+aprint+'\, '+bprint+'\, ln(z/'+str(reference_depth)+')}$')
    # binned mean vs mean depth
    plt.scatter(sliced_mu, np.log10(sliced_mean_depth), color='red', linewidth=3, label='average logscale flux of binned data')
    plt.errorbar(sliced_mu, np.log10(sliced_mean_depth), xerr = sliced_st_dev, capsize = 7, capthick=2, elinewidth=1.5, ls='none', color='red', linewidth=3)
    plt.gca().invert_yaxis()
    plt.xlabel("Binned average of Ln (POC flux [mg $\mathregular{ m^{-2} day^{-1}}$])", **hfont)
    plt.ylabel("$\mathregular{Log_{10}}$ (Depth [m])", **hfont)
    plt.legend(loc="lower right", fontsize=16)
    
    plt.savefig('_binned_scatterplot.pdf', format='pdf')
    plt.show()
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
"""MODEL OVERLAY FUNCTIONS"""
    
def plot_basic_scatterplot(data, x_bounds=[15, 4000]):
    """
    plot each data point on a realspace scatterplot and a logspace scatterplot.
    """
    xd=np.linspace(x_bounds[0], x_bounds[1],100)
    
    depth = data[:,8].astype(float)
    flux = data[:,17].astype(float)
    y = [[np.log(f)] for f in flux]
    x = [[np.log(d/500)] for d in depth]

    plt.rcParams["figure.figsize"] = (20,20)
    plt.rcParams["axes.labelsize"] = 25
    hfont = {'fontname':'Times New Roman'}
    plt.rcParams["xtick.labelsize"] = 24
    plt.rcParams["ytick.labelsize"] = 24
    plt.rcParams["font.family"] = "Times New Roman"
    
    plt.subplot(2, 2, 1)
    plt.yticks(fontname = "Times New Roman")
    plt.xticks(fontname = "Times New Roman")

    plt.scatter(y, depth, color='black', alpha=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel("Ln (POC flux [mg $\mathregular{ m^{-2} day^{-1}}$])", **hfont)
    plt.ylabel("Depth [m]", **hfont)

    plt.subplot(2, 2, 2)
    plt.yticks(fontname = "Times New Roman")
    plt.xticks(fontname = "Times New Roman")

    plt.scatter(np.exp(y), depth, color='black', alpha=0.5)

    plt.xlim((-20,400))
    plt.gca().invert_yaxis()
    plt.xlabel("POC flux [mg $\mathregular{ m^{-2} day^{-1}}$]", **hfont)
    plt.ylabel("Depth [m]", **hfont)

    plt.savefig('_simple_depth_scatterplot.pdf', format='pdf')
    plt.show()
    
    
    
def plot_full_scatterplot_with_parameters(data, A, B, variance, x_bounds=[15, 4000], reference_depth = 100, display_mean = True):
    """
    given parameters for a lognormal power law model, make a scatterplot of the data and overlay 
    the mean power law and standard deviation intervals
    """
    xd=np.linspace(x_bounds[0], x_bounds[1],100)
    yd=A+B*np.log(xd/reference_depth)
    #exp_yd = np.exp(yd)
    mean_yd = [y + variance/2 for y in yd]
    
    meanaprint = str(round(A+variance, 2))
    aprint = str(round(A, 1))
    bprint = str(round(B, 2))
    varprint = str(round(variance, 2))

    stdev=np.sqrt(variance)
    stdevprint = str(round(stdev, 2))
    twostdevprint = str(round(2*stdev, 2))
    
    depth = data[:,8].astype(float)
    flux = data[:,17].astype(float)
    y = [[np.log(f)] for f in flux]
    x = [[np.log(d/reference_depth)] for d in depth]

    plt.rcParams["figure.figsize"] = (20,20)
    plt.rcParams["axes.labelsize"] = 25
    hfont = {'fontname':'Times New Roman'}
    plt.rcParams["xtick.labelsize"] = 24
    plt.rcParams["ytick.labelsize"] = 24
    plt.rcParams["font.family"] = "Times New Roman"

    
    
    plt.subplot(2, 2, 1)
    plt.yticks(fontname = "Times New Roman")
    plt.xticks(fontname = "Times New Roman")
    plt.fill_betweenx(xd, yd - stdev - stdev, yd + stdev+stdev,
                     color=(0.7,0.7,1), label = '$\pm$'+twostdevprint)

    plt.fill_betweenx(xd, yd - stdev, yd + stdev,
                     color=(0.5,0.5,1), label = '$\pm$'+stdevprint)


    plt.scatter(y, depth, color='black', alpha=0.5)
    plt.plot(yd, xd, color='green', linewidth=3, label = '$\mathregular{'+aprint+'\,'+bprint+'\, ln(z/'+str(reference_depth)+')}$')
    if display_mean:
        plt.plot(mean_yd, xd, color='brown', linewidth=3, label = '$\mathregular{'+meanaprint+'\,'+bprint+'\, ln(z/'+str(reference_depth)+')}$')
    plt.gca().invert_yaxis()
    plt.xlabel("Ln (POC flux [mg $\mathregular{ m^{-2} day^{-1}}$])", **hfont)
    plt.ylabel("Depth [m]", **hfont)
    plt.legend(loc="lower right", fontsize=24)
    
    
    
    plt.subplot(2, 2, 2)
    plt.yticks(fontname = "Times New Roman")
    plt.xticks(fontname = "Times New Roman")

    plt.fill_betweenx(xd, np.exp(yd - stdev-stdev), np.exp(yd + stdev+stdev),
                     color=(0.7,0.7,1), label = '$\pm$'+twostdevprint)

    plt.fill_betweenx(xd, np.exp(yd - stdev), np.exp(yd + stdev),
                     color=(0.5,0.5,1), label = '$\pm$'+stdevprint)

    plt.scatter(np.exp(y), depth, color='black', alpha=0.5)
    plt.plot(np.exp(yd), xd, color='green', linewidth=3, label = '$\mathregular{exp(\,'+aprint+'\,'+bprint+' \,ln(z/'+str(reference_depth)+')\,)}$')
    
    if display_mean:
        plt.plot(np.exp(mean_yd), xd, color='brown', linewidth=3, label = '$\mathregular{exp(\,'+meanaprint+'\,'+bprint+'\, ln(z/'+str(reference_depth)+')\,)}$')

    plt.xlim((-20,400))
    plt.gca().invert_yaxis()
    plt.xlabel("POC flux [mg $\mathregular{ m^{-2} day^{-1}}$]", **hfont)
    plt.ylabel("Depth [m]", **hfont)
    plt.legend(loc="lower right", fontsize=24)
    plt.xlim((-20,400))

    plt.savefig('_scatterplot_model_overlay.pdf', format='pdf')
    plt.show()
    
    
    
# no longer using
def plot_full_scatterplot_comparison(data, A, B, variance, data_2, A_2, B_2, variance_2, x_bounds=[50, 4000]):
    """
    broken right now but the idea is to compare two figure 2s to each other
    """
    xd=np.linspace(x_bounds[0], x_bounds[1],100)
    yd=A+B*np.log(xd/500)
    yd_2=A_2+B_2*np.log(xd/500)

    aprint = str(round(A, 1))
    bprint = str(round(B, 2))
    varprint = str(round(variance, 2))
    aprint_2 = str(round(A_2, 1))
    bprint_2 = str(round(B_2, 2))
    varprint_2 = str(round(variance_2, 2))

    stdev=np.sqrt(variance)
    stdev_2=np.sqrt(variance_2)
    
    depth = data[:,8].astype(float)
    flux = data[:,17].astype(float)
    y = [[np.log(f)] for f in flux]
    x = [[np.log(d/500)] for d in depth]
    depth_2 = data_2[:,8].astype(float)
    flux_2 = data_2[:,17].astype(float)
    y_2 = [[np.log(f)] for f in flux_2]
    x_2 = [[np.log(d/500)] for d in depth_2]

    plt.rcParams["figure.figsize"] = (20,10)
    plt.rcParams["axes.labelsize"] = 25
    hfont = {'fontname':'Times New Roman'}
    plt.rcParams["xtick.labelsize"] = 24
    plt.rcParams["ytick.labelsize"] = 24
    plt.rcParams["font.family"] = "Times New Roman"

    plt.subplot(1, 2, 1)
    plt.yticks(fontname = "Times New Roman")
    plt.xticks(fontname = "Times New Roman")
    
    # plot original model
    plt.fill_betweenx(xd, yd - stdev - stdev, yd + stdev+stdev,
                     color=(0.7,0.7,1), label = '$\pm$ 2 st. dev.')

    plt.fill_betweenx(xd, yd - stdev, yd + stdev,
                     color=(0.5,0.5,1), label = '$\pm$ 1 st. dev.')
    
    plt.scatter(y, depth, color=(0,0,0.5), alpha=0.5)
    plt.plot(yd, xd, color='orange', linewidth=3, label = '$\mathregular{'+aprint+'\,'+bprint+'\, ln(z/500)}$')
    
    # plot model 2
    plt.fill_betweenx(xd, yd_2 - stdev_2 - stdev_2, yd_2 + stdev_2+stdev_2,
                     color=(1,0.7,0.7), label = '$\pm$ 2 st. dev.')

    plt.fill_betweenx(xd, yd_2 - stdev_2, yd_2 + stdev_2,
                     color=(1,0.5,0.5), label = '$\pm$ 1 st. dev.')


    plt.scatter(y_2, depth_2, color=(0.5, 0, 0), alpha=0.5)
    plt.plot(yd_2, xd, color='green', linewidth=3, label = '$\mathregular{'+aprint_2+'\,'+bprint_2+'\, ln(z/500)}$')
    
    plt.gca().invert_yaxis()
    plt.xlabel("Ln (POC flux [mg $\mathregular{ m^{-2} day^{-1}}$])", **hfont)
    plt.ylabel("Depth [m]", **hfont)
    plt.legend(loc="lower right", fontsize=24)

    plt.subplot(1, 2, 2)
    plt.yticks(fontname = "Times New Roman")
    plt.xticks(fontname = "Times New Roman")

    plt.fill_betweenx(xd, np.exp(yd - stdev-stdev), np.exp(yd + stdev+stdev),
                     color=(0.7,0.7,1), label = '$\pm$ 2 st. dev.')

    plt.fill_betweenx(xd, np.exp(yd - stdev), np.exp(yd + stdev),
                     color=(0.5,0.5,1), label = '$\pm$ 1 st. dev.')

    plt.scatter(np.exp(y), depth, color='black', alpha=0.5)
    plt.plot(np.exp(yd), xd, color='green', linewidth=3, label = '$\mathregular{exp(\,'+aprint+'\,'+bprint+' \,ln(z/500)\,)}$')

    plt.xlim((-20,400))
    plt.gca().invert_yaxis()
    plt.xlabel("POC flux [mg $\mathregular{ m^{-2} day^{-1}}$]", **hfont)
    plt.ylabel("Depth [m]", **hfont)
    plt.legend(loc="lower right", fontsize=24)

    plt.show()

    
# no longer using
def plot_scatterplot_comparison(data, data_2, x_bounds=[50, 4000]):
    """
    just plots two distributions
    """
    
    depth = data[:,8].astype(float)
    flux = data[:,17].astype(float)
    y = [[np.log(f)] for f in flux]
    x = [[np.log(d/500)] for d in depth]
    depth_2 = data_2[:,8].astype(float)
    flux_2 = data_2[:,17].astype(float)
    y_2 = [[np.log(f)] for f in flux_2]
    x_2 = [[np.log(d/500)] for d in depth_2]

    plt.rcParams["figure.figsize"] = (20,10)
    plt.rcParams["axes.labelsize"] = 25
    hfont = {'fontname':'Times New Roman'}
    plt.rcParams["xtick.labelsize"] = 24
    plt.rcParams["ytick.labelsize"] = 24
    plt.rcParams["font.family"] = "Times New Roman"

    plt.subplot(1, 2, 1)
    plt.yticks(fontname = "Times New Roman")
    plt.xticks(fontname = "Times New Roman")
    
    # plot original dataset
    plt.scatter(y, depth, color=(0,0,0.7), alpha=0.5)
        
    # plot dataset 2
    plt.scatter(y_2, depth_2, color=(0.7,0,0), alpha=0.5)
        
    plt.gca().invert_yaxis()
    plt.xlabel("Ln (POC flux [mg $\mathregular{ m^{-2} day^{-1}}$])", **hfont)
    plt.ylabel("Depth [m]", **hfont)
    
    plt.subplot(1, 2, 2)
    plt.yticks(fontname = "Times New Roman")
    plt.xticks(fontname = "Times New Roman")

    plt.scatter(np.exp(y), depth, color=(0,0,0.7), alpha=0.5)
    
    plt.scatter(np.exp(y_2), depth_2, color=(0.7,0,0), alpha=0.5)

    plt.gca().invert_yaxis()
    plt.xlabel("POC flux [mg $\mathregular{ m^{-2} day^{-1}}$]", **hfont)
    plt.ylabel("Depth [m]", **hfont)

    plt.show()
    

    
    
    
    
    
    
    
    
    
    
    
"""Z SCORE PLOTTING FUNCTIONS"""

def plot_zscore_hist(data, A, B, variance, reference_depth = 100):
    """
    given data and a power law model, make z scores based on model residuals and plot them on a histogram
    """
    depth = data[:,8].astype(float)
    flux = data[:,17].astype(float)
    zscores = [(np.log(flux[i])-(A+B*np.log(depth[i]/reference_depth)))/(np.sqrt(variance)) for i in range(len(flux))]
    
    print('mean z-score:', sum(zscores)/len(zscores))
    print('st. dev. of z-scores:', np.sqrt(np.var(zscores)))

    plt.rcParams["figure.figsize"] = (10,10)
    plt.rcParams["axes.labelsize"] = 25
    hfont = {'fontname':'Times New Roman'}
    plt.rcParams["xtick.labelsize"] = 24
    plt.rcParams["ytick.labelsize"] = 24
    plt.rcParams["font.family"] = "Times New Roman"

    plt.xlabel("Z-score", **hfont)
    plt.ylabel("Probability density", **hfont)
    plt.yticks(fontname = "Times New Roman")
    plt.xticks(fontname = "Times New Roman")

    plt.hist(zscores, bins=45, color='blue',alpha=0.5,density=True)
    mu=0
    sigma=1
    x = np.linspace(mu - 3.5*sigma, mu + 3.5*sigma, 500)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), color='black', linewidth=3)
    
    plt.savefig('_zscore_hist.pdf', format='pdf')
    plt.show()