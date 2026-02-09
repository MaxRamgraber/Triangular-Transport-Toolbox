if __name__ == '__main__':    

    # Load in a number of libraries we will use
    import numpy as np
    import scipy.stats
    import copy
    from triangular_transport_toolbox import transport_map
    import matplotlib.pyplot as plt
    
    # Import shared utilities
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import lorenz_dynamics, rk4
    
    # Close all open figures
    plt.close("all")
    
    # In this example file, we will consider data assimilation for a chaotic,
    # three-dimensional dynamical system known as Lorenz-63. Using triangular
    # transport methods for filtering results in an algorithm we call an 
    # Ensemble Transport Filter (EnTS), a nonlinear generalization of the well-
    # known Ensemble Kalman Filter (EnKF.)
    
    # =========================================================================
    # Set up the exercise
    # =========================================================================
    
    # Set a random seed
    np.random.seed(0)
    
    # Define problem dimensions
    O                   = 3 # Observation space dimensions
    D                   = 3 # State space dimensions
    
    # Ensemble size
    N                   = 500
    
    # Time-related parameters
    T                   = 1000  # Full time series length
    dt                  = 0.1   # Time step length
    dti                 = 0.05  # Time step increment
    
    # Observation error
    obs_sd              = 2
    
    # In this study, we introduce regularization for the transport map.
    lmbda               = 0.05
    
    # Maximum polynomial order for EnTF / EnTS; Feel free to change this value
    maxorder_filter     = 3
        
    #%%
    # =========================================================================
    # Generate synthetic observations
    # =========================================================================
   
    # Create the synthetic reference
    synthetic_truth         = np.zeros((T,1,D))
    synthetic_truth[0,0,:]  = scipy.stats.norm.rvs(size=3)
    
    for t in np.arange(0,T-1,1):
         
        # Make a Lorenz forecast
        synthetic_truth[t+1,:,:] = rk4(
            Z           = copy.copy(synthetic_truth[t,:,:]),
            fun         = lorenz_dynamics,
            t           = 0,
            dt          = dti,
            nt          = int(dt/dti))
        
    # Remove the unnecessary particle index
    synthetic_truth     = synthetic_truth[:,0,:]
        
    # Create observations
    observations        = synthetic_truth + scipy.stats.norm.rvs(scale = obs_sd, size = synthetic_truth.shape)
        
    #%%
    # =========================================================================
    # Set up the transport map object
    # =========================================================================
            
    # Define the map component functions. The target distribution over which we
    # do inference is six-dimensional, because we assume we observe all three
    # variable dimensions (three state observations + three states). The graph
    # of the system looks like:
    #   
    #         ┌---------------┐
    #         |               |
    #        x_a --- x_b --- x_c
    #         |       |       |
    #        y_a     y_b     y_c
    #
    # A straightforward assimilation scheme would form a six-dimensional target
    # distribution, and condition everything at once. If we want to exploit the
    # conditional independence structure in this graph, we can subdivide this
    # update into three separate operations, each assimilating one observation:
    #   
    #            Operation 1
    #         ┌---------------┐
    #         |      (3)      |         --> sample new y_b
    #    (2) x_a --- x_b --- x_c (4)
    #         | 
    #    (1) y_a  
    #
    #            Operation 2
    #         ┌---------------┐
    #         |      (2)      |         --> sample new y_c
    #    (3) x_a --- x_b --- x_c (4)
    #                 |        
    #                y_b        
    #                (1)
    #
    #            Operation 3
    #         ┌---------------┐
    #         |      (4)      |
    #    (3) x_a --- x_b --- x_c (2)
    #                         |
    #                        y_c (1)
    #
    # We apply the three conditioning operations in sequence, each conditioning
    # on one observation at a time. After each update, we have to sample new
    # observation predictions. The numbers next to each node define the their
    # position in the triangular map.
    
    
    # For the later inference problem, we are only interested in extracting 
    # conditionals of this six-dimensional distribution. As established in 
    # previous exercises, this means we only have to define the lower three map
    # component functions - those corresponding to the three states.
    if maxorder_filter == 1: # Map is linear
        
        # We can explot a bit of sparsity in each graph
        nonmonotone_filter  = [
            [[],[0]],
            [[],    [1]],
            [[],    [1],[2]]]
    
        monotone_filter     = [
            [[1]],
            [[2]],
            [[3]]]
        
    else: # Map is nonlinear
        
        # We use combinations of linear terms and Hermite functions for the 
        # non-monotone terms
        nonmonotone_filter  = [
            [[],[0]]+[[0]*od+['HF'] for od in np.arange(1,maxorder_filter+1,1)],
            [[],[1]]+[[1]*od+['HF'] for od in np.arange(1,maxorder_filter+1,1)],
            [[],[1]]+[[1]*od+['HF'] for od in np.arange(1,maxorder_filter+1,1)]+[[2]]+[[2]*od+['HF'] for od in np.arange(1,maxorder_filter+1,1)]]
   
        # Here, we use map component functions of differing complexity. We use
        # nonlinear combinations of edge terms and integrated radial basis 
        # functions for the first state, corresponding to the observed state,
        # and linear terms for all lower dependencies.
        monotone_filter     = [
            ['LET 1']+['iRBF 1']*(maxorder_filter-1)+['RET 1'],
            [[2]],
            [[3]]]
    
    
    # If a previous transport map object exists, delete it
    if "tm" in globals():
        del tm

    # Let's create the transport map object. We initiate it with dummy random
    # variables for now.
    tm     = transport_map(
        monotone                = monotone_filter,
        nonmonotone             = nonmonotone_filter,
        X                       = np.random.uniform(size=(N,1+D)), # Dummy input
        polynomial_type         = "hermite function",
        monotonicity            = "separable monotonicity",
        regularization          = "l2",
        regularization_lambda   = lmbda,
        verbose                 = False)
    
    #%%
    # =========================================================================
    # Ensemble Transport Filtering
    # =========================================================================
    
    # Create an empty list for the ensemble mean RMSEs
    RMSE_list   = []
    
    # Let's set up an empty array for the samples obtained during filtering
    Xt          = np.zeros((T,N,D))
    
    # Draw initial samples from a standard Gaussian
    Xt[0,...]   = scipy.stats.norm.rvs(size=(N,D))
    
    # Create a copy of this array for the forecast. We don't really need this 
    # array for this example, but it will be important in the next example.
    Xft         = copy.copy(Xt)
    
    # Start the filtering
    for t in np.arange(0,T,1):
        
        # Print the update
        print('Timestep '+str(t+1).zfill(4)+'|'+str(T).zfill(4))
        
        # Copy the forecast into the analysis array
        Xt[t,...]   = copy.copy(Xft[t,...])
        
        # Lets assimilate these observations one-at-a-time
        for idx,perm in enumerate([[0,1,2],[1,0,2],[2,1,0]]):
            
            # Before the update step, Ensemble Transport Filters must sample  
            # the stochastic observation model. Sample independent errors for 
            # each state.
            Yt = copy.copy(Xt[t,:,:][:,idx]) + \
                scipy.stats.norm.rvs(
                    loc     = 0,
                    scale   = obs_sd,
                    size    = (N))
            
            # Concatenate observations and samples into six-dimensional samples 
            # from the target density
            map_input = copy.copy(np.column_stack((
                Yt[:,np.newaxis],       # First O dimensions:   simulated observations
                Xt[t,:,:][:,perm])))    # Next D dimensions:    predicted states
            
            # Reset the transport map for the new values
            tm.reset(map_input)
            
            # Optimize the transport map
            tm.optimize()
            
            # ---------------------------------------------------------------------
            # Implement the composite map update
            # ---------------------------------------------------------------------
             
            # For the composite map, we use reference samples from  the pushforward
            # samples
            Z_pushforward = tm.map(map_input)
            
            # Create an array with replicated of the observations
            Y_star = np.repeat(
                a       = observations[t,idx].reshape((1,1)),
                repeats = N, 
                axis    = 0)
            
            # Apply the inverse map
            ret = tm.inverse_map(
                X_star      = Y_star,
                Z           = Z_pushforward)
            
            # Undo the permutation of the states
            ret = ret[:,perm]
            
            # Save the result in the analysis array
            Xt[t,...]   = copy.copy(ret)
        
        # ---------------------------------------------------------------------
        # Detemrine RMSE, make a forecast to the next timestep
        # ---------------------------------------------------------------------
         
        # Calculate ensemble mean RMSE
        RMSE = (np.mean(Xt[t,...],axis=0) - synthetic_truth[t,:])**2
        RMSE = np.mean(RMSE)
        RMSE = np.sqrt(RMSE)
        RMSE_list.append(RMSE)
        
        # After the analysis step, make a forecast to the next timestep
        if t < T-1:
            
            # Make a Lorenz forecast
            Xft[t+1,:,:] = rk4(
                Z           = copy.copy(Xt[t,:,:]),
                fun         = lorenz_dynamics,
                t           = 0,
                dt          = dti,
                nt          = int(dt/dti))
        
    # Plot the results
    plt.figure(figsize=(7,7))
    plt.plot(RMSE_list,color='xkcd:grey')
    plt.xlabel('timestep')
    plt.ylabel('ensemble mean RMSE')
    plt.title('EnTF order '+str(maxorder_filter)+' | RMSE: '+"{:.3f}".format(np.mean(RMSE_list)))
    plt.savefig('01_RMSE_EnTF_order='+str(maxorder_filter)+'.png')