if __name__ == '__main__':    

    # Load in a number of libraries we will use
    import numpy as np
    import scipy.stats
    import copy
    from transport_map import *
    import time
    import pickle
    import matplotlib.pyplot as plt
    
    # Close all open figures
    plt.close("all")
    
    # In this example file, we will build on the foundation established by the
    # Ensemble Transport Filter, and demonstrate how to extend these principles
    # to smoothing. Specifically, we consider a backward Ensemble Transport
    # Smoother, which is a nonlinear generalization of the Ensemble Rauch-Tung-
    # Striebel Smoother.    
    
    # =========================================================================
    # Set up the Lorenz-63 dynamics
    # =========================================================================
    
    # First, we must define the system dynamics and the time integration scheme
    
    # Lorenz-63 dynamics
    def lorenz_dynamics(t, Z, beta=8/3, rho=28, sigma=10):
        
        if len(Z.shape) == 1: # Only one particle
        
            dZ1ds   = - sigma*Z[0] + sigma*Z[1]
            dZ2ds   = - Z[0]*Z[2] + rho*Z[0] - Z[1]
            dZ3ds   = Z[0]*Z[1] - beta*Z[2]
            
            dyn     = np.asarray([dZ1ds, dZ2ds, dZ3ds])
            
        else:
            
            dZ1ds   = - sigma*Z[...,0] + sigma*Z[...,1]
            dZ2ds   = - Z[...,0]*Z[...,2] + rho*Z[...,0] - Z[...,1]
            dZ3ds   = Z[...,0]*Z[...,1] - beta*Z[...,2]
    
            dyn     = np.column_stack((dZ1ds, dZ2ds, dZ3ds))
    
        return dyn
    
    # Fourth-order Runge-Kutta scheme
    def rk4(Z,fun,t=0,dt=1,nt=1):#(x0, y0, x, h):
        
        """
        Parameters
            t       : initial time
            Z       : initial states
            fun     : function to be integrated
            dt      : time step length
            nt      : number of time steps
        
        """
        
        # Prepare array for use
        if len(Z.shape) == 1: # We have only one particle, convert it to correct format
            Z       = Z[np.newaxis,:]
            
        # Go through all time steps
        for i in range(nt):
            
            # Calculate the RK4 values
            k1  = fun(t + i*dt,           Z);
            k2  = fun(t + i*dt + 0.5*dt,  Z + dt/2*k1);
            k3  = fun(t + i*dt + 0.5*dt,  Z + dt/2*k2);
            k4  = fun(t + i*dt + dt,      Z + dt*k3);
        
            # Update next value
            Z   += dt/6*(k1 + 2*k2 + 2*k3 + k4)
        
        return Z
    
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
    maxorder_filter     = 3     # Maximum map order for the filter
    maxorder_smoother   = 3     # Maximum map order for the smoother
        
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
    observations        = copy.copy(synthetic_truth) + scipy.stats.norm.rvs(scale = obs_sd, size = synthetic_truth.shape)
        
    #%%
    """
    |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    The backward EnTS is built  on the foundation of a previous EnTF run. The
    lines below are an exact replicate of example_05.py. Feel free to skip past
    it if you have already familiarized yourself with the previous example.
    |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    """
    
    
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
        print('Filtering: timestep '+str(t+1).zfill(4)+'|'+str(T).zfill(4))
        
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
    
    """
    |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    We resume the smoothing algorithm at this point.
    |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    """
    
#%%

    # =========================================================================
    # Backward smoothing
    # =========================================================================
            
    # The backward smoother conditions a joint distribution of states at 
    # adjacent times on smoothed states passed down backward along the graph.
    # As such, the graph for the smoothing update has six nodes, one for the 
    # three states at time s and at time s+1, and is dense, that is to say 
    # there is no conditional independence to exploit. As this map differs in
    # structure from the one we used in filtering, we must create a new map.
    
    # Define the map component functions
    if maxorder_smoother == 1: # Linear smoother
        
        nonmonotone_BWS  = [
            [[],[0],[1],[2]],
            [[],[0],[1],[2],[3]],
            [[],[0],[1],[2],[3],[4]]]
    
        monotone_BWS     = [
            [[3]],
            [[4]],
            [[5]]]
        
    else: # Nonlinear smoother
        
        nonmonotone_BWS  = [
            [[],[0]]+[[0]*od+['HF'] for od in np.arange(1,maxorder_smoother+1,1)] + [[1]]+[[1]*od+['HF'] for od in np.arange(1,maxorder_smoother+1,1)] + [[2]]+[[2]*od+['HF'] for od in np.arange(1,maxorder_smoother+1,1)],
            [[],[0]]+[[0]*od+['HF'] for od in np.arange(1,maxorder_smoother+1,1)] + [[1]]+[[1]*od+['HF'] for od in np.arange(1,maxorder_smoother+1,1)] + [[2]]+[[2]*od+['HF'] for od in np.arange(1,maxorder_smoother+1,1)] + [[3]]+[[3]*od+['HF'] for od in np.arange(1,maxorder_smoother+1,1)],
            [[],[0]]+[[0]*od+['HF'] for od in np.arange(1,maxorder_smoother+1,1)] + [[1]]+[[1]*od+['HF'] for od in np.arange(1,maxorder_smoother+1,1)] + [[2]]+[[2]*od+['HF'] for od in np.arange(1,maxorder_smoother+1,1)] + [[3]]+[[3]*od+['HF'] for od in np.arange(1,maxorder_smoother+1,1)] + [[4]]+[[4]*od+['HF'] for od in np.arange(1,maxorder_smoother+1,1)]]

        # Even for the nonlinear backward smoother, we keep the monotone terms
        # linear. This reduces computational demand by quite a bit, and often
        # still performs quite well.
        monotone_BWS     = [
            [[3]],
            [[4]],
            [[5]]]
    
    # Delete the pre-existing map object
    if "tm" in globals():
        del tm

    # Parameterize the transport map
    tm     = transport_map(
        monotone                = monotone_BWS,
        nonmonotone             = nonmonotone_BWS,
        X                       = np.random.uniform(size=(N,int(2*D))), 
        polynomial_type         = "probabilist's hermite",  
        monotonicity            = "separable monotonicity",
        regularization          = "l2",
        regularization_lambda   = lmbda,
        verbose                 = False)    
    
    # =========================================================================
    # Now start the actual backward smoother
    # =========================================================================
    
    # Let's set up an empty array for the samples obtained during smoothing. We
    # initiate it as a copy of the filtering analysis samples, but really we
    # only need the last marginal of the filtering samples (Xt[-1,...])-
    Xst         = copy.copy(Xt)
    
    # Create a list for the ensemble mean RMSEs. The first smoothing marginal
    # is also the last filtering marginal, so they share an RMSE value.
    RMSE_list_smoother = [RMSE_list[-1]]
    
    # Go backward through time
    for t in range(T-2,-1,-1):
        
        # Print the update
        print('Smoothing: timestep '+str(t+1).zfill(4)+'|'+str(T).zfill(4))
        
        # The joint distributions conditioned by the backward smoother are 
        # constructed from filtering analysis and forecast samples obtained 
        # during the preceding filtering pass.
        map_input = copy.copy(np.column_stack((
            Xft[t+1,:,:],   # First D dimensions:   filtering forecast
            Xt[t,...])))    # Next D dimensions:    filtering analysis
    
        # Reset the map
        tm.reset(copy.copy(map_input))
        
        # Start optimizing the transport map
        tm.optimize()
        
        # ---------------------------------------------------------------------
        # Composite map
        # ---------------------------------------------------------------------
        
        # We condition on previous smoothing samples
        X_star = copy.copy(Xst[t+1,...])
        
        # The composite map uses reference samples 
        # from an application of the forward map
        Z_pushforward = tm.map(map_input)
        
        # Invert / Condition the map
        ret = tm.inverse_map(
            X_star      = X_star,
            Z           = Z_pushforward)

        # Copy the results into the smoothing array
        Xst[t,...]      = copy.copy(ret)

        # Calculate RMSE
        RMSE            = (np.mean(Xst[t,...],axis=0) - synthetic_truth[t,:])**2
        RMSE            = np.mean(RMSE)
        RMSE            = np.sqrt(RMSE)
        RMSE_list_smoother.append(RMSE)
        
    # Plot the results
    plt.figure(figsize=(7,7))
    plt.plot(RMSE_list,color='xkcd:grey',label='Ensemble Transport Filter')
    plt.plot(np.flip(RMSE_list_smoother),color='xkcd:orangish red',label='Ensemble Transport Smoother')
    plt.xlabel('timestep')
    plt.ylabel('ensemble mean RMSE')
    plt.legend()
    plt.title('EnTF order '+str(maxorder_filter)+' | RMSE: '+"{:.3f}".format(np.mean(RMSE_list))+' | EnTs order '+str(maxorder_smoother)+' | RMSE: '+"{:.3f}".format(np.mean(RMSE_list_smoother)))
    plt.savefig('02_RMSE_EnTS_order='+str(maxorder_filter)+'_smoother_order='+str(maxorder_smoother)+'.png')