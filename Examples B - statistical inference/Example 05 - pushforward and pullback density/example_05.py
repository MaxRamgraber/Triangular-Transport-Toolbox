import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib
import os
from transport_map import *

# =============================================================================
# Step 1: Train the map
# =============================================================================

# In this example, we will evaluate the pushforward density (the map's 
# approximation to the standard Gaussian reference distribution) and the 
# pullback density (the map's approximation to the target distribution).

np.random.seed(0)

plt.close('all')

# To begin, let us define a function that samples a 2D target pdf
def sample_wavy_distribution(size):

    X       = np.zeros((size,2))
    X[:,0]  = (scipy.stats.beta.rvs(
        a       = 2,
        b       = 2,
        size    = size)*2-1)*3
    X[:,1]  = scipy.stats.norm.rvs(
        scale   = 1/6,
        size    = size)
    
    X[:,1]  += np.sin(X[:,0]*1.2)
    
    X[:,0] /= 1.5
    X[:,1] *= 1.5
    
    return X

# Here is the corresponding function for its log density
def density_wavy_distribution(X):
    
    from scipy.spatial import KDTree
    import scipy.stats
    import numpy as np
    import copy
    
    # Scale X
    X       = copy.copy(X)
    X[:,0]  *= 1.5
    X[:,1]  /= 1.5
    
    X[:,1]  -= np.sin(X[:,0]*1.2)
    
    locX    = (X[:,0]/3+1)/2
    locX[np.where(locX < 0.000001)] = 0.000001
    locX[np.where(locX > 0.999999)] = 0.999999
    
    logpdf  = np.log(1/6)
    logpdf  += scipy.stats.beta.logpdf(
        x       = locX,
        a       = 2,
        b       = 2)
    logpdf  += scipy.stats.norm.logpdf(
        x       = X[:,1],
        scale   = 1/6)
    
    return logpdf

# Training ensemble size
N   = 1000

# Draw that many samples
X   = sample_wavy_distribution(N)

# Create empty lists for the map component specifications
monotone    = []
nonmonotone = []

# Specify the map components
for k in range(2):
    
    # Create empty lists for the map component specifications
    monotone    = []
    nonmonotone = []
    
    # Specify the map components
    for k in range(2):
        
        # Level 1: One list entry for each dimension (two in total)
        monotone.append([])
        nonmonotone.append([[]]) # An empty list "[]" denotes a constant
    
        # Go through each polynomial order
        for o in range(3):
            
            # Nonmonotone part ----------------------------------------------------
            
            if k > 0:
                # Level 2: Specify the polynomial order of this term;
                # It's a Hermite function term
                if o == 0:
                    nonmonotone[-1].append([k-1]*(o+1))
                else:
                    nonmonotone[-1].append([k-1]*(o+1)+['HF'])
            
            # Monotone part -------------------------------------------------------
            
            # Level 2: Specify the polynomial order of this term;
            if o == 0:
                monotone[-1].append([k])
            else:
                monotone[-1].append(
                    'iRBF '+str(k))

# Delete any map object which might already exist
if "tm" in globals():
    del tm

# Parameterize the transport map
tm     = transport_map(
    monotone                = monotone,
    nonmonotone             = nonmonotone,
    X                       = copy.copy(X),         # Training ensemble
    polynomial_type         = "hermite function",   # We use Hermite functions for stability
    monotonicity            = "separable monotonicity", # Because we have cross-terms, we require the integrated rectifier formulation
    standardize_samples     = True,                 # Standardize X before training
    workers                 = 1,                    # Number of workers for the parallel optimization; 1 is not parallel
    quadrature_input        = {                     # Keywords for the Gaussian quadrature used for integration
        'order'         : 25,       # If the map is bad, increase this number; takes more computational effort
        'adaptive'      : False,
        'threshold'     : 1E-9,
        'verbose'       : False,
        'increment'     : 6})

# Optimize the map
tm.optimize()

# =============================================================================
# Step 2: Evaluate the pushforward and pullback densities
# =============================================================================

# Create an evaluation grid
X_grid,Y_grid = np.meshgrid(
    np.linspace(-3,3,101),
    np.linspace(-3,3,101) )

# Convert the grid into a N-by-D array
XY_grid     = np.column_stack((
    np.ndarray.flatten(X_grid),
    np.ndarray.flatten(Y_grid) ))

# Evaluate the pushforward density
pushforward_density = tm.evaluate_pushforward_density(
    Z               = XY_grid, 
    log_target_pdf  = density_wavy_distribution)

# Evaluate the pullback density
pullback_density = tm.evaluate_pullback_density(
    X               = XY_grid)

#%%

# =============================================================================
# Step 3: Plot the results
# =============================================================================

# Plot the results
plt.figure(figsize=(13,5))

# Create a subplot for the reference
plt.subplot(1,2,1)

# Draw the contours of the reference pdf
plt.contour(
    X_grid,
    Y_grid,
    scipy.stats.multivariate_normal.pdf(XY_grid, mean = np.zeros(2), cov = np.identity(2)).reshape((101,101)),
    cmap = "turbo")
plt.colorbar()

# Generate samples from the reference
Z_true = scipy.stats.multivariate_normal.rvs(
    mean = np.zeros(2), 
    cov = np.identity(2),
    size = N)

# Plot the samples
plt.scatter(
    Z_true[:,0],
    Z_true[:,1],
    s = 3,
    color = "xkcd:dark grey",
    label = "reference samples")

# Add a legend
plt.legend()

# Equalize the axes
plt.axis("equal")

# Add the title
plt.title("Reference samples and reference pdf")


# Create a subplot for the pushforward
plt.subplot(1,2,2)

# Draw the contours of the pushforward density
plt.contour(
    X_grid,
    Y_grid,
    pushforward_density.reshape((101,101)),
    cmap = "turbo")
plt.colorbar()

# Sample the pushforward density via the forward map
Z_pushforward = tm.map(X)

# Plot the pushforward samples
plt.scatter(
    Z_pushforward[:,0],
    Z_pushforward[:,1],
    s = 3,
    color = "xkcd:dark grey",
    label = "pushforward samples")

# Add a legend
plt.legend()

# Equalize the axes
plt.axis("equal")

# Add a title
plt.title("Pushforward samples and pushforward pdf")

plt.savefig('pushforward_density.pdf',dpi=600,bbox_inches='tight')
plt.savefig('pushforward_density.png',dpi=600,bbox_inches='tight')

#%%

# Plot the results
plt.figure(figsize=(13,5))

# Create a subplot for the target
plt.subplot(1,2,1)

# Draw the contours of the unnormalized target pdf
plt.contour(
    X_grid,
    Y_grid,
    np.exp(density_wavy_distribution(XY_grid)).reshape((101,101)),
    cmap = "turbo")
plt.colorbar()

# Plot the training samples
plt.scatter(
    X[:,0],
    X[:,1],
    s = 3,
    color = "xkcd:dark grey",
    label = "training samples")

# Add a legend
plt.legend()

# Equalize the axes
plt.axis("equal")

# Add the title
plt.title("Training samples and unnormalized target pdf")


# Create a subplot for the pushforward
plt.subplot(1,2,2)

# Draw the contours of the pushforward density
plt.contour(
    X_grid,
    Y_grid,
    pullback_density.reshape((101,101)),
    cmap = "turbo")
plt.colorbar()

# Sample the pullback density via the inverse map
X_pullback = tm.inverse_map(Z_true)

# Plot the pushforward samples
plt.scatter(
    X_pullback[:,0],
    X_pullback[:,1],
    s = 3,
    color = "xkcd:dark grey",
    label = "pullback samples")

# Add a legend
plt.legend()

# Equalize the axes
plt.axis("equal")

# Add a title
plt.title("Pullback samples and pullback pdf")

plt.savefig('pullback_density.pdf',dpi=600,bbox_inches='tight')
plt.savefig('pullback_density.png',dpi=600,bbox_inches='tight')
