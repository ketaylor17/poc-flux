from scipy import io
from scipy import optimize
from scipy import special as sc
from scipy import stats
from scipy.stats import t
import numpy as np
import pandas as pd
from numpy.random import default_rng
import numpy.random as random
import csv
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import LinearRegression
import math
from astropy import stats as astrostats
import random












"""DATA CLEANING FUNCTIONS"""

# regardless of the source they came from, data points with these labels for their trap type are reliable.
# need to look at individual trap type and not just studies, since some studies use a combination 
# of moored traps, drifting traps, and/or thorium.
reliable_trap_type_descriptions = ['Drifter, type not specified', 
                   'cylindrical particle interceptor trap free floating array'
                   'Indented rotary sphere time series trap',
                   'Drifting Sediment Traps Technicap PPS 5',
                   'Particle interceptor trap',
                   'drifting sediment trap array, KC Denmark model 28.200',
                   'drifting',
                   'free floating particle interceptor trap, acrylic tube',
                   'neutrally buoyant sediment trap, PELAGRA',
                   'Clap Trap',
                   'Free flaoting trap, type not specified',
                   'NBST 15',
                   'NBST 17',
                   'NBST 14',
                   'NBST 16',
                   'Drifter',
                   'NBST 11',
                   'NBST 12',
                   'NBST 13',
                   'Particle interceptor trap, type not specified' #from torres-valdes, think this is jgofs
                   ]

# reliable_reference_ids = [33, 11, 14, 9, 10, 38, 39, 43, 41, 32, 44, 42, 3, 4, 5, 6, 24, 20, 13]

# ID numbers of sources that can be trusted above 1000m
# need this in addition to a list of reliable trap types since some studies didn't have a trap type listed
# but still turned out to be reliable
reliable_trap_reference_ids = [11, 14, 9, 10, 3, 4, 5, 6, 24, 20, 13]

# no longer using thorium data - this doesn't matter
# not sure about 38 and 39
thorium_reference_ids = [33,38,39,41,43,32,44,42]

def load_data():
    """
    return numpy array of all data with values for poc and depth
    """
    with open('GO_flux.tab') as f:
        reader = csv.reader(f, delimiter="\t")
        dataset = list(reader)
    desc = dataset[87] # row 87 describes the data in all the columns
    print(desc) # shows which columns contain which data
    data = dataset[88:15880] # 88 to the end is the actual data
    poc_data = np.array(data)[:,17] # column 17 is poc, so poc_data is all poc measurements
    #depth_data = np.array(data)[:,8] # column 8 is depth, so depth_data is all depth measurements

    ind = [x != '' for x in poc_data] # a filter for existence of poc measurements
    data = np.array(data)
    print('size of raw dataset:', len(data))
    data = data[ind]
    print('number of observations with values for poc:', len(data))
    return data
    

def filter_data(data, *funcs):
    """
    return numpy array of only those data entries which satisfy all functions in *funcs
    """
    for func in funcs:
        ind = [func(x) for x in data]
        data = data[ind]
    return data

def reference_filter(func):
    """
    takes a function and applies it to reference type (column 0)
    """
    return lambda x: func(x[0].astype(int))

def trap_type_filter(func):
    """
    takes a function and applies it to trap type (column 5)
    """
    return lambda x: func(x[5])

def depth_filter(func):
    """
    takes a function and applies it to depth (column 8)
    """
    return lambda x: func(x[8].astype(float))

def filter_by_traptype_and_reference(data, trap_types, reference_ids, cutoff_depth):
    """
    return numpy array of those data entries strictly shallower than cutoff_depth with a trap type in 
    trap_types or a reference id in reference_ids
    """
    depth_filt = depth_filter(lambda x: x<cutoff_depth)
    trap_filt = trap_type_filter(lambda x: x in trap_types)
    ref_filt = reference_filter(lambda x: x in reference_ids)
    return filter_data(data, lambda x: depth_filt(x), lambda x: (trap_filt(x) or ref_filt(x)))

def get_reliable_trap_data(data, cutoff_depth):
    """
    return numpy array of those data entries strictly above cutoff_depth with a reliable trap type.
    reliable = from a study/reference which we know uses all reliable traps, i.e., id is in reliable_trap_reference_ids.
    OR reliable = from any study and has a reliable trap type, i.e., trap name is in reliable_trap_type_descriptions.
    need to look at individual trap type and not just studies since some studies use a combination 
    of moored traps, drifting traps, and/or thorium.
    """
    depth_filt = depth_filter(lambda x: x<cutoff_depth)
    trap_filt = trap_type_filter(lambda x: x in reliable_trap_type_descriptions)
    ref_filt = reference_filter(lambda x: x in reliable_trap_reference_ids)
    return filter_data(data, lambda x: depth_filt(x), lambda x: (trap_filt(x) or ref_filt(x)))

# no longer using thorium data - this doesn't matter
def get_thorium_data(data, cutoff_depth):
    """
    return numpy array of thorium measurements strictly above cutoff_depth 
    """
    depth_filt = depth_filter(lambda x: x<cutoff_depth)
    thorium_filt = reference_filter(lambda x: x in thorium_reference_ids)
    return filter_data(data, lambda x: depth_filt(x), lambda x: thorium_filt(x))














"""BINNED SCATTERPLOT / BINNED MODEL FUNCTIONS"""

def slice_it(flux, depth, boundaries):
    """
    split data into bins with given boundaries and return summary stats.
    the bin defined by depth1 < depth2 is exclusive at depth1 (shallower) and inclusive at depth2 (deeper)
    """
    sliced_flux=[]
    sliced_depth=[]
    sliced_mu=[]
    sliced_variance=[]
    i = 0
    while i+1<len(boundaries):
        ind = [d>boundaries[i] and d<=boundaries[i+1] for d in depth]
        sliced_flux.append(flux[ind])
        sliced_depth.append(depth[ind])
        sliced_mu.append(sum(np.log(flux[ind]))/len(flux[ind]))
        sliced_variance.append(np.var(np.log(flux[ind])))
        i = i+1
    sliced_mean_depth = [sum(x)/len(x) for x in sliced_depth]
    
    # all of these variables are arrays of length equal to the number of bins = len(boundaries) - 1
    return (sliced_flux, sliced_depth, sliced_mu, sliced_variance, sliced_mean_depth, boundaries)

# no longer using, since we decided to bin on logscale
def slice_lin(flux, depth, slice_len):
    """
    split data into bins of depth slice_len and return summary stats
    """
    boundaries=[0]
    while boundaries[-1]<4000:
        boundaries.append(boundaries[-1]+slice_len)
    return slice_it(flux, depth, boundaries)


def slice_log(flux, depth, slice_len, scale_factor):
    """
    split data into bins that increase by a factor scale_factor and return summary stats
    """
    boundaries=[0,slice_len]
    while boundaries[-1]<4000:
        boundaries.append(boundaries[-1]*scale_factor)
    return slice_it(flux, depth, boundaries)
    

# no longer using, since we decided to bin on logscale
def slice_count(flux, depth, num_points):
    """
    split data into bins with num_points points in them and return summary stats
    """
    # shuffle points so that points at the same depth are randomly allocated to upper/lower bin
    biglist = (np.transpose([flux, depth]).tolist())
    random.shuffle(biglist)
    biglist.sort(key=lambda x:x[1])
    
    flux = np.array(biglist)[:,0]
    depth = np.array(biglist)[:,1]
    
    sliced_flux=[]
    sliced_depth=[]
    sliced_mu=[]
    sliced_variance=[]
    i = 0
    while len(depth)>i:
        sliced_flux.append(flux[i:min(len(depth),i+num_points)])
        sliced_depth.append(depth[i:min(len(depth),i+num_points)])
        sliced_mu.append(sum(np.log(flux[i:min(len(depth),i+num_points)]))/len(flux[i:min(len(depth),i+num_points)]))
        sliced_variance.append(np.var(np.log(flux[i:min(len(depth),i+num_points)])))
        i = i+num_points
        
    sliced_mean_depth = [sum(x)/len(x) for x in sliced_depth]
    return (sliced_flux, sliced_depth, sliced_mu, sliced_variance, sliced_mean_depth)


def mean_power_law_params(sliced_mu, sliced_variance, sliced_mean_depth, reference_depth = 100):
    """
    regression for mean power law. returns A, B, and 95% confidence intervals.
    y = A + B*x 
    logmean = A + B*log(depth/reference_depth)
    logmean of distribution is a function of depth.
    this function is intended to use on binned data. it's a regression on logmean flux per bin vs. mean depth per bin.
    ("logmean flux" = mean of the logs of a list of fluxes, see function slice_it above)
    """
    x = np.log([d/reference_depth for d in sliced_mean_depth])
    y = sliced_mu
    
    # standardized t test 95% confidence interval
    ts=t.interval(.95, len(x)-2, loc=0, scale=1)
    
    # simple linear regression
    result = stats.linregress(x, y)
    A = result.intercept
    A_std_err = result.intercept_stderr
    # 95% confidence interval for A
    A_int = A+np.multiply(ts,A_std_err)
    
    B = result.slope
    B_std_err = result.stderr
    # 95% confidence interval for B
    B_int = B+np.multiply(ts,B_std_err)
    
    # constant variance
    # it would also be reasonable to calculate this as the variance of points from the curve defined by A and B
    # note: this is variance in logspace, i.e. average of squared errors of the form (logmean - log(flux))^2
    var = sum(sliced_variance)/len(sliced_variance)
    
    return {"A":A, "A_std_err":A_std_err, "A_interval":A_int, "B":B, "B_std_err":B_std_err, "B_interval":B_int, "variance":var}


def make_binned_fit_data(data, bin_type, slice_len=None, scale_factor=None, num_points=None, boundaries=None):
    """
    bin the data and calculate mean flux, mean depth, variance of each bin.
    also calculates and returns the best fit power law to these binned average values.
    """
    depth = data[:,8].astype(float) # depth column from the dataset
    flux = data[:,17].astype(float) # poc column from the dataset
    (sliced_flux, sliced_depth, sliced_mu, sliced_variance, sliced_mean_depth, boundaries) = (None, None, None, None, None, None)
    
    if bin_type == "custom":
        (sliced_flux, sliced_depth, sliced_mu, sliced_variance, sliced_mean_depth, boundaries) = slice_it(flux, depth, boundaries)
    elif bin_type == "linear":
        (sliced_flux, sliced_depth, sliced_mu, sliced_variance, sliced_mean_depth, boundaries) = slice_lin(flux, depth, slice_len)
    elif bin_type == "point count":
        (sliced_flux, sliced_depth, sliced_mu, sliced_variance, sliced_mean_depth) = slice_count(flux, depth, num_points)
    elif bin_type == "logscale":
        (sliced_flux, sliced_depth, sliced_mu, sliced_variance, sliced_mean_depth, boundaries) = slice_log(flux, depth, slice_len,
                                                                                               scale_factor)
    power_law_params = mean_power_law_params(sliced_mu, sliced_variance, sliced_mean_depth)
    
    return (sliced_flux, sliced_depth, sliced_mu, sliced_variance, sliced_mean_depth, power_law_params, boundaries)

















"""UNBINNED LINEAR REGRESSION FUNCTIONS"""

def lin_reg_power_law(data, reference_depth = 100):
    """
    transform the data into logspace and do a linear regression. that's it!
    same as the binned model if each point got its own bin.
    """
    depth = data[:,8].astype(float)
    flux = data[:,17].astype(float)
    
    y = [[np.log(f)] for f in flux]
    x = [[np.log(d/reference_depth)] for d in depth]
    xy = list(zip(x,y))
    
    reg = LinearRegression(fit_intercept=True).fit(x, y)
    B=reg.coef_[0][0]
    A=reg.intercept_[0]
    sqr_res = np.power(A+B*np.array(x)-np.array(y), 2)
    var = sum(s[0] for s in sqr_res)/len(sqr_res)
    
    return [A, B, var]
    
# no longer using
def make_iterated_parameter_model(data, num_iters):
    """
    do figure 2, the iterated weighting scheme. more useful for nonconstant variance
    """
    depth = data[:,8].astype(float)
    flux = data[:,17].astype(float)
    
    # transform data appropriately: y=ln(flux), x=ln(depth/500)
    # y=A+B*x
    # constant variance
    y = [[np.log(f)] for f in flux]
    x = [[np.log(d/500)] for d in depth]
    xy = list(zip(x,y))
    
    param_list = []
    
    for i in range(num_iters):
        this_xy = random.choices(xy, k=len(xy))
        this_x, this_y = zip(*this_xy)
        reg = LinearRegression(fit_intercept=True).fit(this_x, this_y)
        B=reg.coef_[0][0]
        A=reg.intercept_[0]
        sqr_res = np.power(A+B*np.array(this_x)-np.array(this_y), 2)
        var = sum(s[0] for s in sqr_res)/len(sqr_res)
        param_list.append((A,B,var))
    
    A_list, B_list, var_list = zip(*param_list)
    
    
    return [(sum(A_list)/num_iters, np.var(A_list)), (sum(B_list)/num_iters, np.var(B_list)), 
            (sum(var_list)/num_iters, np.var(var_list)), param_list]

















"""Z SCORE ANALYSIS FUNCTIONS"""

def z_score(A, B, var, depth, flux, reference_depth = 100):
    """
    given z and corresponding depth, returns (logflux-mu)/sigma
    where mu=A+B*np.log(depth/reference_depth)
    """
    z = (np.log(flux)-(A+B*np.log(depth/reference_depth)))/(np.sqrt(var))
    return z
















# no longer using - linear regression alone produces a gaussian-looking distribution, so no need to get fancy
"""GAUSSIAN SHAPE BLACK MAGIC PARAMETER FITTING FUNCTIONS"""

# no longer using
def astrokuiper(zeta):
    return astrostats.kuiper(zeta, stats.norm.cdf)[0]

# no longer using
def spearman(zeta, depth):
    r = stats.spearmanr(zeta, depth).correlation
    if math.isnan(r):
        return 1
    else:
        return abs(r)

# no longer using
def abs_spearman(zeta, depth):
    r = stats.spearmanr(abs(zeta), depth).correlation
    if math.isnan(r):
        return 1
    else:
        return abs(r)

import time

# no longer using
def error_func_inner(A, B, var, depth, flux):
    """
    calculate error term sum
    """
    #print('getting zeta')
    #t1 = time.time()
    zeta = z_score(A, B, var, depth, flux)
    #print(time.time()-t1)
    #print('getting spmn')
    #t1 = time.time()
    error_spmn = spearman(zeta, depth)
    #print(time.time()-t1)
    #print('getting abs spmn')
    #t1 = time.time()
    error_abs_spmn = abs_spearman(zeta, depth)
    #print(time.time()-t1)
    #print('getting mean')
    #t1 = time.time()
    error_mean = abs(sum(zeta)/len(zeta))
    #print(time.time()-t1)
    #print('getting stdev')
    #t1 = time.time()
    error_stdev = abs(np.sqrt(np.var(zeta))-1)
    #print(time.time()-t1)
    #print('getting kuiper')
    #t1 = time.time()
    error_kuiper = astrokuiper(zeta)
    #print(time.time()-t1)

    return error_spmn+error_abs_spmn+error_mean+error_stdev+error_kuiper

# no longer using
def get_parameter_error(A, B, var, data):
    depth = np.array(data[:,8].astype(float))
    flux = np.array(data[:,17].astype(float))
    return error_func_inner(A, B, var, depth, flux)

# no longer using
def min_error(A_range, B_range, var_range, data, print_it=False):
    """
    iterate over all parameter ranges to find parameter combination with the lowest error
    """
    
    depth = np.array(data[:,8].astype(float))
    flux = np.array(data[:,17].astype(float))

    params = []
    min_val = error_func_inner(A_range[0], B_range[0], var_range[0], depth, flux)
    
    for a in A_range:
        for b in B_range:
            for v in var_range:
                err = error_func_inner(a,b,v, depth, flux)
                if err == min_val:
                    params.append([a,b,v])
                if err < min_val or np.isnan(min_val):
                    params = [[a,b,v]]
                    min_val = err
            if print_it:
                print(str(params)+'    '+str([a,b,v]))
    return (params, min_val)

# no longer using
def iterated_min_error(A_range, B_range, var_range, data, num_iters, print_it=False):
    """
    do the error function sampling at random from dataset
    """
    results_list = []
    depth = np.array(data[:,8].astype(float))
    flux = np.array(data[:,17].astype(float))
    depflux = list(zip(depth,flux))
    for _ in range(num_iters):
        this_depflux = random.choices(depflux, k=len(depflux))
        this_depth, this_flux = zip(*this_depflux)
        this_depth = np.array(this_depth)
        this_flux = np.array(this_flux)

        params = []
        min_val = error_func_inner(A_range[0], B_range[0], var_range[0], this_depth, this_flux)

        for a in A_range:
            for b in B_range:
                for v in var_range:
                    err = error_func_inner(a,b,v, this_depth, this_flux)
                    if err == min_val:
                        params.append((a,b,v))
                    if err < min_val or np.isnan(min_val):
                        params = [(a,b,v)]
                        min_val = err
                if print_it:
                    print(str(params)+'    '+str((a,b,v)))
        if len(params)>1: 
            print('multiple equivalent parameter combos exist:', params)
        results_list.append(params[0])
        
    A_list, B_list, var_list = zip(*results_list)
    return [(sum(A_list)/num_iters, np.var(A_list)), (sum(B_list)/num_iters, np.var(B_list)), 
            (sum(var_list)/num_iters, np.var(var_list)), results_list]