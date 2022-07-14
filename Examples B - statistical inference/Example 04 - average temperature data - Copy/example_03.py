# Load a number of libraries required for this exercise
import scipy.stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy
import itertools
import os
import pickle
import re

# Load in the transport map class
from transport_map import *

# Find the current working directory
root_directory = os.path.dirname(os.path.realpath(__file__))

# Set a random seed
np.random.seed(0)

# Close all open figures
plt.close('all')

# =============================================================================
# Step 1: Preparing the data
# =============================================================================

# For this example, we use daily average temperature data in Munich, Germany, 
# and Moscow, Russia, to demonstrate a data-based example for statistical
# inference. The data sets we used are extracted from here:
# https://academic.udayton.edu/kissock/http/Weather/default.htm

# Let's load in the data for Munich
with open('DLMUNICH.txt') as text_file:
    lines = text_file.read().splitlines()

# Go through all lines, store the date and the daily average temperature
data_Munich     = []
times           = []
for line in lines:
    chunks      = line.split()
    data_Munich .append(float(chunks[-1]))
    times       .append(chunks[2]+'-'+chunks[1]+'-'+chunks[0])

# Now let's load the data for Moscow
with open('RSMOSCOW.txt') as text_file:
    lines = text_file.read().splitlines()

# Likewise go through every line of the text file. Only add a data point if we
# have an average temperature for both Munich and Moscow on the same date.
data        = []

# Go through every line of the Moscow data file
for line in lines:
    
    # Split the string
    chunks    = line.split()
    
    # Do we have a temperature data point for Munich?
    if chunks[2]+'-'+chunks[1]+'-'+chunks[0] in times:
        
        # In which index is that data point?
        idx     = times.index(chunks[2]+'-'+chunks[1]+'-'+chunks[0])
        
        # Are both data points valid measurements, or NaNs?
        if data_Munich[idx] > -99 and float(chunks[-1]) > -99:
            
            # If yes, append a new data point
            data    .append([data_Munich[idx],float(chunks[-1])])
        
# Convert the list of lists into a N-by-2 array, with temperatures in Munich in
# the first column, and temperatures for Moscow in the second column
data        = np.asarray(data)

# Convert the temperatures from Fahrenheit to Celsius
data        = (data - 32) * .5556

# These data points constitute samples from our target density function
X           = data

# Plot the data
plt.figure(figsize=(7,7))
plt.scatter(X[:,0],X[:,1],s=1,color='xkcd:grey',label='real data')
plt.xlabel('daily average temperature in Munich, Germany (°C)')
plt.ylabel('daily average temperature in Moscow, Russia (°C)')
plt.legend()
plt.savefig('temperature_data.png')


#%%
# =============================================================================
# Define the transport map parameterization
# =============================================================================

# Once more, we have to define the map parameterization. We have seen this in 
# earlier examples.

# Create empty lists for the map component specifications
monotone    = []
nonmonotone = []

# Change the maxorder, and see how the map approximation function changes
maxorder    = 10

# Here, we try  different form of map parameterization. Let's try using maps
# with separable monotonicity. These are often much more efficient, but do not
# allow for cross-terms or nonmonotone basis functions in the 'monotone' list.
for k in range(X.shape[-1]):
    
    # Level 1: Add an empty list entry for each map component function
    monotone.append([])
    nonmonotone.append([]) # An empty list "[]" denotes a constant
    
    # Level 2: We initiate the nonmonotone terms with a constant
    nonmonotone[-1].append([])

    # Nonmonotone part --------------------------------------------------------

    # Go through every polynomial order
    for order in range(maxorder):
        
        # We only have non-constant nonmonotone terms past the first map 
        # component, and we already added the constant term earlier, so only do
        # this for the second map component function (k > 0).
        if k > 0: 
            
            # The nonmonotone basis functions can be as nonmonotone as we want.
            # Hermite functions are generally a good choice.
            nonmonotone[-1].append([k-1]*(order+1)+['HF'])
            
    # Monotone part -----------------------------------------------------------
    
    # Let's get more fancy with the monotone part this time. If the order  we 
    # specified is one, then use a linear term. Otherwise, use a few monotone 
    # special functions: Left edge terms, integrated radial basis functions, 
    # and right edge terms
    
    # The specified order is one
    if maxorder == 1:
        
        # Then just add a linear term
        monotone[-1].append([k])
        
    # Otherweise, the order is greater than one. Let's use special terms.
    else:
        
        # Add a left edge term. The order matters for these special terms. 
        # While they are placed according to marginal quantiles, they are 
        # placed from left to right. We want the left edge term to be left.
        monotone[-1].append('LET '+str(k))
                
        # Lets only add maxorder-1 iRBFs
        for order in range(maxorder-1):
            
            # Add an integrated radial basis function
            monotone[-1].append('iRBF '+str(k))
    
        # Then add a right edge term 
        monotone[-1].append('RET '+str(k))
        

#%%
# =============================================================================
# Create the transport map object
# =============================================================================

# With the map parameterization (nonmonotone, monotone) defined and the target
# samples (X) obtained, we can start creating the transport map object.

# To begin, delete any map object which might already exist.
if "tm" in globals():
    del tm

# Create the transport map object tm
tm     = transport_map(
    monotone                = monotone,                 # Specify the monotone parts of the map component function
    nonmonotone             = nonmonotone,              # Specify the nonmonotone parts of the map component function
    X                       = X,                        # A N-by-D matrix of training samples (N = ensemble size, D = variable space dimension)
    polynomial_type         = "hermite function",       # What types of polynomials did we specify? The option 'Hermite functions' here are re-scaled probabilist's Hermites, to avoid numerical overflow for higher-order terms
    monotonicity            = "separable monotonicity", # Are we ensuring monotonicity through 'integrated rectifier' or 'separable monotonicity'?
    standardize_samples     = True,                     # Standardize the training ensemble X? Should always be True
    verbose                 = True,                     # Shall we print the map's progress?
    workers                 = 1)                        # Number of workers for the parallel optimization.)


# This map is cheap to optimize.
tm.optimize()

    
#%%
# =============================================================================
# Generative modelling
# =============================================================================

# Let's do something interesting. Let's generate some more, fake pairs of daily
# temperatures between Munich and Moscow. By learning the target distribution
# underlying the data set, we can use the inverse transport map to generate 
# new, approximate realizations of the target. 

# Let's start by drawing new samples from the standard Gaussian reference.
Z       = scipy.stats.norm.rvs(size=(1000,2))

# Now pull these samples back to the target
X_gen   = tm.inverse_map(Z)

# Plot the results
plt.figure(figsize=(7,7))
plt.scatter(X[:,0],X[:,1],s=1,color='xkcd:grey',label='real data')
plt.scatter(X_gen[:,0],X_gen[:,1],s=2,color='xkcd:orangish red',label='generated data')
plt.xlabel('daily average temperature in Munich, Germany (°C)')
plt.ylabel('daily average temperature in Moscow, Russia (°C)')
plt.legend()
plt.savefig('generative_modeling.png')


#%%
# =============================================================================
# Conditional sampling
# =============================================================================

# We can also use transport maps to answer statistical questions about the 
# joint target density function. For example, we can wonder what temperatures
# we can expect in Moscow if temperatures in Munich are at a certain value.
# This requires conditional sampling. 

# Let's start by drawing new samples from the standard Gaussian reference. As
# in the earlier exercises, we only need reference samples for the second 
# dimension, so these samples will be a column vector rather than a matrix with
# two columns.
Z       = scipy.stats.norm.rvs(size=(1000,1))

# For what temperatures in Munich to we want to investigate the temperatures in
# Moscow? Feel free to adjust this value
X_star  = np.ones((1000,1))*(-5) # Let's try -5 °C first.

# Now pull these samples back to the target
X_cond  = tm.inverse_map(Z = Z, X_star = X_star)

# Now plot the results
plt.figure(figsize=(8.5,7))
gs  = matplotlib.gridspec.GridSpec(
    nrows           = 1,
    ncols           = 2, 
    width_ratios    = [1.0,0.2], 
    wspace          = 0.)
plt.subplot(gs[0,0])
plt.scatter(X[:,0],X[:,1],s=1,color='xkcd:grey',label='real data',zorder=5)
plt.scatter(X_cond[:,0],X_cond[:,1],s=2,color='xkcd:orangish red',label='conditional temperatures in Moscow',zorder=10,alpha=0.1)
plt.xlabel('daily average temperature in Munich, Germany (°C)')
plt.ylabel('daily average temperature in Moscow, Russia (°C)')
plt.legend()
ylims = plt.gca().get_ylim()
plt.subplot(gs[0,1])
plt.hist(X_cond[:,1],orientation='horizontal',bins=30,color='xkcd:orangish red')
plt.ylim(ylims)
plt.gca().yaxis.tick_right()
plt.gca().set_ylabel('conditional temperature in Moscow')
plt.gca().yaxis.set_label_position("right")
plt.savefig('conditional_temperatures_Moscow.png')
plt.show()




