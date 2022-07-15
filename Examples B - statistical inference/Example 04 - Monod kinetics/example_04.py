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
# Step 1: Load the data
# =============================================================================

# In this example, we will show how to use maps for Bayesian inference. Our 
# first example considers parameter inference for Monod kinetics. This model 
# has two parameters we seek to learn: r_max and K
#
#   r_max       - maximum reaction rate
#   K           - half-velocity constant
#
# In this example, we assume that we observe reaction rates for different 
# concentrations of substrate, and seek to find the parameters which best match
# our observations.

# To start off, let us load the data
text_file = open("model_monod.dat", "r")
lines = text_file.readlines()
text_file.close()

# Convert data into arrays
C       = np.zeros(len(lines)-1) # Concentrations for which we make observations
obs_rate= np.zeros(len(lines)-1) # The observed rate
for idx,line in enumerate(lines): # Go through each line string
    if idx > 0: # Ignore the first line - it's only headers
        # Split the string into parts, delimited by tabulators
        split_string = lines[idx].split('\t')
        # The second entry is the concentration, convert it from a string to a number
        C[idx-1] = float(split_string[1])
        # The third entry is the observed rate, convert it from a string to a number
        obs_rate[idx-1] = float(split_string[2])
        
# Let's plot these results
plt.figure(figsize=(7,7))
plt.title('observations')
plt.plot(C,obs_rate,marker='x',color='xkcd:orangish red')
plt.xlabel('substrate concentration')
plt.ylabel('observed rate')
plt.savefig('01_data.png')
        
# =============================================================================
# Step 2: Sampling the prior
# =============================================================================
        
# In the next step, let us define the prior for each of the parameters, then 
# draw samples from it. 

# Let's begin by defining the sample size we will use in these experiments
N       = 1000

# Now draw N samples from the prior for each parameter
r_max   = np.exp(scipy.stats.norm.rvs(loc = 1.5,     scale = 0.5,    size = N))
K       = np.exp(scipy.stats.norm.rvs(loc = 1.0,     scale = 0.5,    size = N))
    
# Let's see how well our prior fits the observations. This is the model for the
# Monod kinetics:
def model_monod(r_max,K,C):
    
    # Create an empty array for the simulated concentrations
    sim_rate = np.zeros((len(r_max),len(C)))
    
    # Simulate all samples
    for n in range(len(r_max)):
        sim_rate[n,:]   = (r_max[n]*C)/(K[n]+C)
    
    return sim_rate

# Run the simulations
sim_rate    = model_monod(r_max,K,C)

# From these simulated rates, let's generate observation predictions. These add
# the observation noise to the simulations
pred_rate   = np.zeros(sim_rate.shape)
for n in range(N):
    pred_rate[n,:]  = sim_rate[n,:] + scipy.stats.norm.rvs(scale=0.1,size=len(C))

# Let's plot these results
plt.figure(figsize=(7,7))
for n in range(N):
    if n == 0:
        plt.plot(C,sim_rate[n,:],color='xkcd:silver', zorder = 5, label = 'predicted',alpha=0.1)
    else:
        plt.plot(C,sim_rate[n,:],color='xkcd:silver', zorder = 5, alpha=0.1)
plt.plot(C,obs_rate,marker='x',color='xkcd:orangish red', zorder = 10, label = 'observed')
plt.title('prior predictions')
plt.xlabel('substrate concentration')
plt.ylabel('reaction rate')
plt.ylim(0,plt.gca().get_ylim()[-1])
plt.legend()
plt.savefig('02_prior_simulations.png')
        
#%%
# =============================================================================
# Step 3: Define the transport map parameterization
# =============================================================================

# Once more, we have to define the map parameterization. We have seen this in 
# earlier examples. In this case, our target density function will not be two-
# dimensional, but 22-dimensional: 20 observation prediction dimensions, and
# 2 parameter dimensions.

# Let's define the dimensionality of the target density
D           = pred_rate.shape[-1] + 2

# Create empty lists for the map component specifications
monotone    = []
nonmonotone = []

# Let's keep the map relatively simple for now
maxorder    = 5

# Let us use Hermite functions and integrated radial basis functions, as 
# before. Recall that we only need the lower block of the map, so let's only 
# define the last few map component functions:
for k in np.arange(D-2,D,1):
    
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
            
            # If the order is one, just add it as a linear term
            if order == 0:
                nonmonotone[-1].append([k-1])
            else:
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
# Step 4: Create the transport map object
# =============================================================================

# First let's define the target samples. If we want to extract conditionals of
# the target density function, as in this case, the values on which we want to
# condition must occupy the upper-most entries of the map. In consequence, we 
# assemble the matrix X as follows:
    
X       = np.column_stack((
    pred_rate,              # 20 dims
    r_max[:,np.newaxis],    # 1 dim
    K[:,np.newaxis] ))      # 1 dim

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
# Step 5: Bayesian inference step
# =============================================================================

# We have established the basics of conditional sampling earlier. To condition
# this target density on specific observations, create a matrix of duplicates
# of the observations.
X_star  = np.repeat(
    a       = obs_rate[np.newaxis,:],
    repeats = N,
    axis    = 0)

# Let's use a composite map approach, so we derive our reference samples from a
# forward application of the map
Z       = tm.map(X)

# Now pull these samples back to the target
X_cond  = tm.inverse_map(Z = Z, X_star = X_star)

# Let's extract the updated parameters
r_max_cond  = X_cond[:,0]
K_cond      = X_cond[:,1]

# Run the simulations
sim_rate_cond   = model_monod(r_max_cond,K_cond,C)

# Let's generate the corresponding observation predictions
pred_rate_cond  = np.zeros(sim_rate_cond.shape)
for n in range(N):
    pred_rate_cond[n,:]     = sim_rate_cond[n,:] + scipy.stats.norm.rvs(scale=0.1,size=len(C))

#%%
# =============================================================================
# Step 6: Plot the results
# =============================================================================

# First, let's plot the prior parameters and the posterior parameters
plt.figure(figsize=(7,7))

# Plot prior and posterior parameters
plt.scatter(r_max,K,s=3,color='xkcd:silver',label='prior')
plt.scatter(r_max_cond,K_cond,s=3,color='xkcd:grass green',label='posterior')
plt.scatter(5,2.4,marker='x',color='xkcd:orangish red',label='truth')

# Complete the figure
plt.title('prior and posterior parameters')
plt.xlabel('maximum reaction rate $r_{max}$')
plt.ylabel('half-velocity constant $K$')
plt.legend()

# Save the figure
plt.savefig('03_posterior_parameters.png')


# Next, let's see how these updated parameters affect the predictions
plt.figure(figsize=(10,10))

# Plot the prior quantiles
q0025   = np.quantile(sim_rate,   q = 0.025, axis = 0)
q0500   = np.quantile(sim_rate,   q = 0.500, axis = 0)
q0975   = np.quantile(sim_rate,   q = 0.975, axis = 0)
plt.plot(C,q0500,color='xkcd:grey',zorder=10,label='prior (median)')
plt.fill(list(C)+list(np.flip(C)),list(q0025)+list(np.flip(q0975)),color='xkcd:grey',alpha=0.25,label='prior (2.5% - 97.5%)')

# Plot the posteiror quantiles
q0025   = np.quantile(sim_rate_cond,   q = 0.025, axis = 0)
q0500   = np.quantile(sim_rate_cond,   q = 0.500, axis = 0)
q0975   = np.quantile(sim_rate_cond,   q = 0.975, axis = 0)
plt.plot(C,q0500,color='xkcd:grass green',zorder=10,label='posterior (median)')
plt.fill(list(C)+list(np.flip(C)),list(q0025)+list(np.flip(q0975)),color='xkcd:grass green',alpha=0.25,label='posterior (2.5% - 97.5%)')

# Plot the observations
plt.plot(C,obs_rate,marker='x',color='xkcd:orangish red', zorder = 10, label = 'observed')

# Complete the figure
plt.title('prior and posterior predictions')
plt.xlabel('substrate concentration')
plt.ylabel('reaction rate')
plt.ylim(0,10)
plt.legend(ncol=3)
plt.savefig('04_posterior_simulations.png')