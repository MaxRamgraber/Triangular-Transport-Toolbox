"""
Triangular transport map toolbox v1.0.0
"""

import numpy as np

class transport_map():
    
    import numpy as np
    import copy
    
    def __init__(self, 
        X, 
        monotone                = None, 
        nonmonotone             = None, 
        polynomial_type         ='hermite function', 
        monotonicity            = 'integrated rectifier', 
        standardize_samples     = True, 
        standardization         = 'standard', 
        workers                 = 1, 
        ST_scale_factor         = 1.0, 
        ST_scale_mode           = 'dynamic', 
        coeffs_init             = 0., 
        alternate_root_finding  = True,
        root_search_truncation  = True,
        verbose                 = True, 
        linearization           = None, 
        linearization_specified_as_quantiles = True, 
        linearization_increment = 1E-6, 
        regularization          = None, 
        regularization_lambda   = 0.1, 
        quadrature_input        = {}, 
        rectifier_type          = 'exponential',
        delta                   = 1E-8,
        adaptation              = False,
        adaptation_map_type     = "cross-terms",
        adaptation_max_order    = 10,
        adaptation_skip_dimensions  = 0,
        adaptation_max_iterations   = 25):
        
        """
        This toolbox contains functions required to construct, optimize, and 
        evaluate transporth methods.
        
        Maximilian Ramgraber, July 2022
        
        
        Variables:
            
            ===================================================================
            General variables
            =================================================================== 
            
            monotone - [default = None]
                [list] : list specifying the structure of the monotone part of 
                the transport map component functions. Required if map 
                adaptation is not used.
                
            nonmonotone  - [default = None]  
                [list] : list specifying the structure of the nonmonotone part 
                of the transport map component functions. Required if map 
                adaptation is not used.
                
            X
                [array] : N-by-D array of the training samples used to optimize
                the transport map, where N is the number of samples and D is 
                the number of dimensions
                
            polynomial_type - [default = 'hermite function']
                [string] : keyword which specifies what kinds of polynomials 
                are used for the transport map component functions. 
                
            monotonicity - [default = 'integrated rectifier']
                [string] : keyword which specifies through what method the 
                transport map ensures monotonicity in the last dimensions. 
                Must be 'integrated rectifier' or 'separable monotonicity'.
            
            standardize_samples - [default = True]
                [boolean] : a True/False flag determining whether the transport
                map should standardize the training samples before optimziation
                
            standardization - [default = 'standard']
                [string] : keyword which specifies whether standardization uses
                mean and standard deviation ('standard') or median and 
                quantiles ('quantiles').
                
            workers - [default = 1]
                [integer] : number of workers for parallel optimization. If set
                to 1, parallelized optimization is inactive. 
                
            ST_scale_factor - [default = 1.0]
                [float] : a float which scales the width of special terms used
                in the map components, such as 'RBF 0', 'iRBF 0', 'LET 0', or
                'RET 0'.
                
            ST_scale_mode - [default = 'dynamic']
                [string] : keyword which defines whether the width of special
                term scale factors is determined based on neighbouring special 
                terms ('dynamic') or fixed as ST_scale_factor ('static').
                
            coeffs_init - [default = 0.]
                [float] : value used to initialize the coefficients at the 
                start of the map optimization.
                
            alternate_root_finding - [default = True]
                [boolean] : flag which determines whether an accelerated, 
                alternate root finding algorithm should be used if monotonicity
                is set to "separable monotonicity". If False, a general 
                bisection algorithm is used instead.
                
            root_search_truncation - [default = True]
                [boolean] : flag which determines whether the root search
                truncates outliers, which serves to prevent numerical overflow
                for flat map tails.
                
            verbose - [default = True]
                [boolean] : a True/False flag which determines whether the map
                prints updates or not. Set to 'False' if running on a cluster 
                to avoid recording excessive output logs.
                
            delta - [default = 1E-8]
                [float] : small increment to prevent numerical underflow. If 
                monotonicity == 'separable monotonicity', provides a small offset
                to  the objective; if monotonicity == 'integrated rectifier', 
                it is added to the rectifier.
                
            Linearization -----------------------------------------------------
                
            linearization - [default = None]
                [float or None] : float which specifies boundary values used
                to linearize the map components in the tails. It's role is 
                specifies by linearization_specified_as_quantiles.
                
            linearization_specified_as_quantiles - [default = True]
                [boolean] : flag which specifies whether the linearization
                thresholds are specifies as quantiles (True) or absolute values
                (False). If True, boundaries are placed at linearization and
                1-linearization, if False, at linearization and -linearization.
                Only used if linearization is not None.
                
            linearization_increment - [default = 1E-6]
                [float] : increment used for the linearization of the map 
                component functions. Only used if linearization is not None.
                
            Regularization ----------------------------------------------------
            
            regularization - [default = None]
                [string or None] : keyword which specifies if regularization is
                used, and if so, what kind of regularization ('L1' or 'L2').
                
            regularization_lambda - [default = 0.1]
                [float] : float which specifies the weight for the map coeff-
                icient regularization. Only used if regularization is not None.
                  
            ===================================================================
            Variables for monotonicity = 'integrated rectifier'
            ===================================================================   
                    
            quadrature_input - [default = {}]
                [dictionary] : dictionary for optional keywords to overwrite
                the default variables in the function Gauss_quadrature. Only 
                used if monotonicity = 'integrated rectifier'.
                
            rectifier_type - [default = 'exponential']
                [string] : keyword which specifies which function is used to 
                rectify the monotone map components. Only used if 
                monotonicity = 'integrated rectifier'.
                
        """
        
        
        import numpy as np
        import copy
        
        # ---------------------------------------------------------------------
        # Load in pre-defined variables
        # ---------------------------------------------------------------------

        # Basis function specification for the monotone and nonmonotone parts 
        # of the map component functions.
        self.monotone               = copy.deepcopy(monotone)
        self.nonmonotone            = copy.deepcopy(nonmonotone)
        
        # How many workers for optimization?
        self.workers                = workers
        
        # Specification for the rectifiers, used when monotonicity is set to
        # 'integrated rectifier'.
        self.rectifier_type         = rectifier_type
        self.delta                  = delta
        self.rect                   = self.rectifier(
            mode        = self.rectifier_type, 
            delta       = self.delta)
        
        # Input for the Gaussian quadrature module, used when monotonicity is 
        # set to 'integrated rectifier'.
        self.quadrature_input       = quadrature_input
        
        # Check if we can pre-calculate the integration points
        if 'xis'        not in list(self.quadrature_input.keys()) and \
           'Ws'         not in list(self.quadrature_input.keys()):
              
            if 'order' not in list(self.quadrature_input.keys()):
                order   = 100   # Default value
            else:
                order   = quadrature_input['order'] # Read input specification
        
            # Weights and integration points are not specified; calculate them
            # To get the weights and positions of the integration points, we must
            # provide the *order*-th Legendre polynomial and its derivative
            # As a first step, get the coefficients of both functions
            coefs       = [0]*order+[1]
            coefs_der   = np.polynomial.legendre.legder(coefs)
            
            # With the coefficients defined, define the Legendre function
            LegendreDer = np.polynomial.legendre.Legendre(coefs_der)
            
            # Obtain the locations of the integration points
            xis = np.polynomial.legendre.legroots(coefs)
            
            # Calculate the weights of the integration points
            Ws  = 2.0/( (1.0-xis**2)*(LegendreDer(xis)**2) )
        
            # Store this result in the dictionary
            self.quadrature_input['xis']    = copy.copy(xis)
            self.quadrature_input['Ws']     = copy.copy(Ws)
        
        # Parameters for special terms optionally used in the specification of
        # the basis functions.
        self.ST_scale_factor        = ST_scale_factor
        self.ST_scale_mode          = ST_scale_mode
        if self.ST_scale_mode not in ['dynamic','static']:
            raise ValueError("'ST_scale_mode' must be either 'dynamic' or 'static'.")
        
        # Is the map being standardized?
        self.standardization        = standardization
        
        # Initial value for the coefficients
        self.coeffs_init            = coeffs_init
        
        # Are we useing alternate root finding?
        self.alternate_root_finding = alternate_root_finding
        
        # If set to True, prevents extrapolation during root finding. Use of
        # this option is not recommended.
        self.root_search_truncation = root_search_truncation
        
        # Should the toolbox print the outputs to the console?
        self.verbose                = verbose
        
        # Are we using regularization?
        self.regularization         = regularization
        self.regularization_lambda  = regularization_lambda
        
        # Are we using linearization?
        self.linearization          = linearization
        self.linearization_specified_as_quantiles     = linearization_specified_as_quantiles
        self.linearization_increment= linearization_increment
        
        # How are we ensuring monotonicity?
        self.monotonicity           = monotonicity
        if self.monotonicity.lower() not in ['integrated rectifier','separable monotonicity']:
            raise ValueError("'monotonicity' type "+str(self.monotonicity)+" not understood. "+\
                "Must be either 'integrated rectifier' or 'separable monotonicity'.")
                
        # ---------------------------------------------------------------------
        # Read and assign the polynomial type
        # ---------------------------------------------------------------------
        
        # What type of polynomials are we using for the specification of the 
        # basis functions in the map component functions?
        self.polynomial_type        = polynomial_type
         
        # Determine the derivative and polynomial terms depending on the chosen type
        if polynomial_type.lower() == 'standard' or polynomial_type.lower() == 'polynomial' or polynomial_type.lower() == 'power series':
            self.polyfunc       = np.polynomial.polynomial.Polynomial
            self.polyfunc_der   = np.polynomial.polynomial.polyder
            self.polyfunc_str   = "np.polynomial.Polynomial"
        elif polynomial_type.lower() == 'hermite' or polynomial_type.lower() == "phycisist's hermite" or polynomial_type.lower() == "phycisists hermite":
            self.polyfunc       = np.polynomial.hermite.Hermite
            self.polyfunc_der   = np.polynomial.hermite.hermder
            self.polyfunc_str   = "np.polynomial.Hermite"
        elif polynomial_type.lower() == 'hermite_e' or polynomial_type.lower() == "probabilist's hermite" or polynomial_type.lower() == "probabilists hermite":
            self.polyfunc       = np.polynomial.hermite_e.HermiteE
            self.polyfunc_der   = np.polynomial.hermite_e.hermeder
            self.polyfunc_str   = "np.polynomial.HermiteE"
        elif polynomial_type.lower() == 'chebyshev':
            self.polyfunc       = np.polynomial.chebyshev.Chebyshev
            self.polyfunc_der   = np.polynomial.chebyshev.chebder
            self.polyfunc_str   = "np.polynomial.Chebyshev"
        elif polynomial_type.lower() == 'laguerre':
            self.polyfunc       = np.polynomial.laguerre.Laguerre
            self.polyfunc_der   = np.polynomial.laguerre.lagder
            self.polyfunc_str   = "np.polynomial.Laguerre"
        elif polynomial_type.lower() == 'legendre':
            self.polyfunc       = np.polynomial.legendre.Legendre
            self.polyfunc_der   = np.polynomial.legendre.legder
            self.polyfunc_str   = "np.polynomial.Legendre"
        elif polynomial_type.lower() == 'hermite function' or polynomial_type.lower() == 'hermite_function' or polynomial_type.lower() == 'hermite functions':
            self.polynomial_type= 'hermite function'    # Unify this polynomial string, so we can use it as a flag
            self.polyfunc       = np.polynomial.hermite_e.HermiteE
            self.polyfunc_der   = np.polynomial.hermite_e.hermeder
            self.polyfunc_str   = "np.polynomial.HermiteE"
        else:
            raise Exception("Polynomial type not understood. The variable polynomial_type should be either 'power series', 'hermite', 'hermite_e', 'chebyshev', 'laguerre', or 'legendre'.")
        
        # ---------------------------------------------------------------------
        # Load and prepare the variables
        # ---------------------------------------------------------------------
        
        # Load and standardize the samples
        self.X                      = copy.copy(X)
        self.standardize_samples    = standardize_samples        
        if self.standardize_samples:
            self.standardize()
           
        # Do we specify map adaptation parameters?
        self.adaptation             = adaptation
        self.adaptation_map_type    = adaptation_map_type.lower()
        self.adaptation_max_order   = adaptation_max_order
        self.adaptation_skip_dimensions = adaptation_skip_dimensions
        self.adaptation_max_iterations  = adaptation_max_iterations
        
        # If we are not adapting the map
        if not self.adaptation:
            
            # Map adaptation is not active       
            self.D                      = len(monotone)
            self.skip_dimensions        = X.shape[-1]-self.D
            
        # If we are adapting the map
        elif self.adaptation:
            
            # Map adaptation is active. Create a linear transport marginal map.
            
            # Define lower map component blocks
            self.D                      = X.shape[-1] - self.adaptation_skip_dimensions
            self.skip_dimensions        = self.adaptation_skip_dimensions
        
            # Initiate dummy monotone and nonmonotone variables
            self.monotone               = []
            self.nonmonotone            = []
            for k in range(self.D):
                self.monotone   .append([[]])
                self.nonmonotone.append([[]])
        
        # ---------------------------------------------------------------------
        # Construct the monotone and non-monotone functions
        # ---------------------------------------------------------------------
        
        # The function_constructor yields six variables:
        #   - fun_mon               : list of monotone functions
        #   - fun_mon_strings       : list of monotone function strings
        #   - coeffs_mon            : list of coefficients for monotone function
        #   - fun_nonmon            : list of nonmonotone functions
        #   - fun_nonmon_strings    : list of nonmonotone function strings   
        #   - coeffs_nonmon         : list of coefficients for nonmonotone function
        
        self.function_constructor_alternative()
        
        # ---------------------------------------------------------------------
        # Precalculate the Psi matrices
        # ---------------------------------------------------------------------
        
        # The function_constructor yields two variables:
        #   - Psi_mon               : list of monotone basis evaluations
        #   - Psi_nonmon            : list of nonmonotone basis evaluations
        
        self.precalculate()
        
        # # Adapt map
        # self.adapt_map()
        
    def adapt_map(self, 
        coeffs              = {}, 
        maxorder_mon        = 10, 
        maxorder_nonmon     = 10,
        threshold_sw        = 0.1,
        threshold_prec      = 0.1,
        sequential_updates  = False,
        map_finished        = None):
        
        """
        This function implements the adaptive transport map algorithm. It is 
        currently only implemented for cross-term integrated maps, and for
        demonstration purposes, so it might be a bit unstable.
        
        =======================================================================
        Variables
        =======================================================================   
                
        increment - [default = 1E-6]
            [float] : the increment for the finite difference approximation to
            the objective function's gradient.
            
        chronicle - [default = False]
            [boolean] : flag which stores the intermediate solutions of the 
            adaptive map algorithm if True. Can be useful to visualize how the 
            map is constructed. Creates a pickled dictionary file name 
            dictionary_adaptation_chronicle.p in the working directory.
        """

        # =====================================================================
        # Separable map adaptation
        # =====================================================================

        if self.adaptation_map_type == 'separable':
            
            import numpy as np
            import scipy.stats
            import copy
            
            # Initiate monotone and nonmonotone terms
            nonmonotone = [[[]] for x in np.arange(self.D)]
            monotone    = [[[x]] for x in np.arange(self.D)]

            # =================================================================
            # Start marginal adaptation
            # =================================================================
            
            # Array with flags for which marginals have been Gaussianized
            Gaussianized    = np.zeros(self.D,dtype=bool)
            
            # Flag to decide when to stop iterating
            iterate         = True
            iteration       = 0
            
            # Create a matrix for the map order
            maporders       = np.zeros((self.D,self.D),dtype=int)
            np.fill_diagonal(maporders,1)
            
            # Create a matrix for the p values of the Shapiro-Wilk test
            pvals_mat       = np.zeros((maxorder_mon,self.D))
            
            while iterate:
                
                # Increase the iteration counter
                iteration   += 1
                
                # -------------------------------------------------------------
                # Reconstruct the new map type
                # -------------------------------------------------------------
                
                # Store the monotone and nonmonotone terms
                self.monotone       = copy.deepcopy(monotone)
                self.nonmonotone    = copy.deepcopy(nonmonotone)
                
                # Re-write the functions
                self.function_constructor_alternative()
                self.precalculate()

                # Optimize the map
                # print(np.arange(self.D)[~Gaussianized])
                self.optimize()
                
                # Apply the map
                Z   = self.map()
                
                # Prepare an array for the pval normality test
                pval_normality_test     = np.zeros(self.D)
                
                # Go through all terms
                for k in range(self.D):

                    # Throw in the p value
                    pval_normality_test[k] = scipy.stats.shapiro(
                        Z[:,k]).pvalue
                    
                # Copy that value into 
                pvals_mat[iteration-1,:] = copy.copy(pval_normality_test)
                
                # for idx in np.where(pval_normality_test < criterion)[0]:
                for idx in np.where(pval_normality_test >= threshold_sw)[0]:
                    index   = np.arange(self.D)[idx]
                    Gaussianized[index] = True
                
                # Increase the map complexity of the non-Gaussian marginals
                for k in np.where(~Gaussianized)[0]:
                    
                    if maporders[k,k+self.skip_dimensions] < maxorder_mon:
                    
                        # Update map complexity storage
                        maporders[k,k+self.skip_dimensions] += 1
                            
                        # Add an integrated iRBF term
                        monotone[k] += ['iRBF '+str(k)]
                    
                    
                if np.sum(Gaussianized) == self.D:
                    iterate     = False
                    
                if iteration >= maxorder_mon-1:
                    iterate     = False

            # =================================================================
            # Start off-diagonal adaptation
            # =================================================================
            
            # Get the standardized precision matrix
            # precmat = np.abs(np.linalg.inv(np.cov(Z.T)))
            covmat  = np.abs(np.cov(Z.T))
            diagval = np.sqrt(np.diag(covmat))
            covmat /= diagval[np.newaxis,:]
            covmat /= diagval[:,np.newaxis]
            
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            #     glasso  = sklearn.covariance.GraphicalLassoCV(max_iter=50).fit(Z)
            # precmat = np.abs(glasso.get_precision())
            
            
            # Get the standardized precision matrix
            precmat = np.abs(np.linalg.inv(np.cov(Z.T)))
            diagval = np.sqrt(np.diag(precmat))
            precmat /= diagval[np.newaxis,:]
            precmat /= diagval[:,np.newaxis]  
            
            # # with warnings.catch_warnings():
            # #     warnings.simplefilter("ignore")
            # #     glasso  = sklearn.covariance.GraphicalLassoCV(max_iter=50).fit(Z)
            # # precmat = np.abs(glasso.get_precision())
            # diagval = np.sqrt(np.diag(precmat))
            # precmat /= diagval[np.newaxis,:]
            # precmat /= diagval[:,np.newaxis]
            
            # Store precmat
            self.covmat     = copy.copy(covmat)
            self.precmat    = copy.copy(precmat)
            
            # Flag to decide when to stop iterating
            iterate         = True
            iteration       = 0
            
            # Create an array to decide when to stop
            if map_finished is None:
                map_finished    = np.zeros((self.D,self.D),dtype=bool)
            precmat_prev    = np.ones((self.D,self.D))
        
            # Store the precision matrix
            precmat_list    = [copy.copy(precmat)]
            
            while iterate:
                
                # Increase the iteration counter
                iteration   += 1
                
                # Store the monotone and nonmonotone terms
                self.monotone       = copy.deepcopy(monotone)
                self.nonmonotone    = copy.deepcopy(nonmonotone)
                
                # Re-write the functions
                self.function_constructor_alternative()
                self.precalculate()
                
                # Optimize the map
                self.optimize()
                
                # Apply the map
                Z   = self.map()
                
                # Attempt to evaluate the precision matrix
                try:
                    
                    if iteration == 1:
                    
                        # Get the standardized precision matrix
                        precmat = np.abs(np.linalg.inv(np.cov(Z.T)))
                        # with warnings.catch_warnings():
                        #     warnings.simplefilter("ignore")
                        #     glasso  = sklearn.covariance.GraphicalLassoCV(max_iter=50).fit(Z)
                        # precmat = np.abs(glasso.get_precision())
                        diagval = np.sqrt(np.diag(precmat))
                        precmat /= diagval[np.newaxis,:]
                        precmat /= diagval[:,np.newaxis]
                        
                    else:
                        
                        # After we found the precision matrix, we proceed with
                        # the correlation matrix for simplicity.
                        precmat = np.corrcoef(Z.T)
                        
                    # Go through all map components
                    for k in range(self.D):
                        
                        # Go through all potential nonmonotone dependencies
                        for j in range(k):
                            
                            # Is there significant correlation
                            # if precmat[k,j] > threshold_prec and precmat[k,j] < precmat_prev[k,j] and not map_finished[k,j]:
                            if precmat[k,j] > threshold_prec and not map_finished[k,j]:
                                
                                # Increase the map complexity by one order
                                maporders[k,j]  += 1
                                
                                # Add the corresponding map component
                                if maporders[k,j] == 1:
                                    nonmonotone[k].append([j]*maporders[k,j])
                                else:
                                    nonmonotone[k].append([j]*maporders[k,j]+['HF'])
                                
                            else:
                                
                                # Mark this map component term as converged
                                map_finished[k,j]   = True
                             
                        # Sort the nonmonotone map components
                        nonmonotone[k].sort()
                    
                    # Store the precision matrix for future reference
                    precmat_prev    = copy.copy(precmat)
                    
                    # And append it to the list
                    precmat_list.append(copy.copy(precmat))
                    
                # If anything failed, stop iterating
                except:
                    
                    # Stop iterating
                    iterate     = False
                    
                # If all lower-triangular entries (excluding diagonal) are 
                # marked as converged, stop iterating
                if np.sum(map_finished) >= self.D*(self.D-1)/2:
                    iterate     = False
                    
                # Raise a warning if the adaptation stopped at the maximum 
                # number of iterations, then stop iterating
                if iteration >= maxorder_nonmon:
                    print("WARNING: Map adaptation stopped at maximum number of iterations.")
                    iterate     = False

            # Store the monotone and nonmonotone terms
            self.monotone       = copy.deepcopy(monotone)
            self.nonmonotone    = copy.deepcopy(nonmonotone)
            
            # Re-write the functions
            self.function_constructor_alternative()
            self.precalculate()
            
            # Optimize the map
            self.optimize()

            # Store the maporders
            self.maporders      = maporders


        # =====================================================================
        # Cross-term map adaptation
        # =====================================================================
        
        elif self.adaptation_map_type == 'cross-terms':
            
            self.adaptation_cross_terms(*coeffs) 
            
        else:
            
            raise Exception("Currently, only adaptation_map_type = 'cross-terms' is implemented.")
        
        
    def check_inputs(self):
        
        """
        This function runs some preliminary checks on the input provided,
        alerting the user to any possible input errors.
        """
        
        if (self.monotone is None or self.nonmonotone is None) and not self.adaptation:
            raise ValueError("Map is undefined. You must either specify 'monotone' and "+\
                "'nonmonotone', or set 'adaptation' = True.")
            
        if self.adaptation_map_type not in ['cross-terms','separable','marginal']:
            raise ValueError("'adaptation_map_type' not understood. Must be either "+\
                "'cross-terms', 'separable', or 'marginal'.")
            
        if self.adaptation_map_type == 'cross-terms' and self.monotonicity.lower() != 'integrated rectifier':
            raise ValueError("It is only possible to use adaptation_map_type = 'cross-terms' if "+\
                "monotonicity = 'integrated rectifier'.")
        
        if self.monotonicity.lower() not in ['integrated rectifier','separable monotonicity']:
            raise ValueError("'monotonicity' type "+str(self.monotonicity)+" not understood. "+\
                "Must be either 'integrated rectifier' or 'separable monotonicity'.")
                
        if self.ST_scale_mode not in ['dynamic','static']:
            raise ValueError("'ST_scale_mode' must be either 'dynamic' or 'static'.")
            
        if self.hermite_function_threshold_mode != 'composite' and \
            self.hermite_function_threshold_mode != 'individual':
                
            raise ValueError("The flag hermite_function_threshold_mode must be " + \
                "'composite' or 'individual'. Currently, it is defined as " + \
                str(self.hermite_function_threshold_mode))
                
        if self.regularization is not None:
            
            if self.monotonicity.lower() != 'separable monotonicity':
            
                if self.regularization.lower() not in ['l2']:
                    raise ValueError("When using 'separable monotonicity'," + \
                        "'regularization' must either be None " + \
                        "(deactivated) or 'L2' (L2 regularization). Currently, it " + \
                        "is defined as "+str(self.regularization))
            else:
                
                if self.regularization.lower() not in ['l2']:
                    raise ValueError("When using 'integrated rectifier'," + \
                        "'regularization' must either be None " + \
                        "(deactivated), 'L1' (L1 regularization) or " + \
                        "'L2' (L2 regularization). Currently, it " + \
                        "is defined as "+str(self.regularization))

    def reset(self,X):
        
        """
        This function is used if the transport map has been initiated with a 
        different set of samples. It resets the standardization variables and
        the map's coefficients, requiring new optimization.
        
        Variables:
        
            X
                [array] : N-by-D array of the training samples used to optimize
                the transport map, where N is the number of samples and D is 
                the number of dimensions
        """
        
        import copy
        
        if len(X.shape) != 2:
            raise Exception('X should be a two-dimensional array of shape (N,D), N = number of samples, D = number of dimensions. Current shape of X is '+str(X.shape))
        
        self.X  = copy.copy(X)
        
        # Standardize the samples, if desired
        if self.standardize_samples:
            self.standardize()

        # Set all parameters to zero
        for k in range(self.D):
            
            # Reset coefficients to zero
            self.coeffs_mon[k]      *= 0
            self.coeffs_nonmon[k]   *= 0
            
            # Provide them with the desired initial values
            self.coeffs_mon[k]      += self.coeffs_init
            self.coeffs_nonmon[k]   += self.coeffs_init
            
        # Precalculate the Psi matrices
        self.precalculate()
    
    def standardize(self):
        
        """
        This function centers the samples around zero and re-scales them to 
        have unit standard deviation. This is important for certain function
        types used in the map component parameterizations, for example Hermite
        functions, which revert to zero farther away from the origin.
        
        The standardization is applied before any other transport operations,
        and reverted before results are returned. It should only affect 
        internal computations.
        """
        
        import numpy as np
        
        # In 'standard' mode, samples are standardized via their mean and 
        # marginal standard deviations
        if self.standardization.lower() == 'standard':
            
            self.X_mean = np.mean( self.X,  axis = 0)
            self.X_std  = np.std(  self.X,  axis = 0)
            
        # In 'quantile' mode, samples are standardized via their quantiles.
        elif self.standardization.lower() == 'quantile' or self.standardization.lower() == 'quantiles':
            
            self.X_mean = np.quantile(self.X,q=0.5,axis=0)
            self.X_std  = (
                np.quantile(self.X-self.X_mean,q=0.8413447460685429,axis=0) - \
                np.quantile(self.X-self.X_mean,q=0.15865525393145707,axis=0) )/2
                
        # Raise an error if the specified mode is not recognized
        else:
            
            raise ValueError("'standardization' must be either 'standard' or 'quantiles'.")
        
        # Standardize the samples
        self.X      -= self.X_mean
        self.X      /= self.X_std
        
    def precalculate(self):
        
        """
        This function pre-calculates matrices of basis function evaluations for
        the samples provided. These matrices can be used to optimize the maps
        more quickly.
        """
        
        import copy
        
        # Precalculate locations of any special terms
        self.determine_special_term_locations()
        
        # Prepare precalculation matrices
        self.Psi_mon    = []
        self.Psi_nonmon = []
        
        # Add the monotone basis function's derivatives if we use separable monotonicity
        if self.monotonicity.lower() == 'separable monotonicity':
            self.der_Psi_mon= []
        
        # Precalculate matrices
        for k in range(self.D):
            
            # Evaluate the basis functions
            self.Psi_mon    .append(copy.copy(self.fun_mon[k](copy.copy(self.X),self)))
            self.Psi_nonmon .append(copy.copy(self.fun_nonmon[k](copy.copy(self.X),self)))
            
            # Add the monotone basis function's derivatives if we use separable monotonicity
            if self.monotonicity.lower() == 'separable monotonicity':
                self.der_Psi_mon.append(copy.copy(self.der_fun_mon[k](copy.copy(self.X),self)))
              
        return
    
    def write_basis_function(self,term,mode='standard',k=None):
        
        """
        This function assembles a string for a specific term of the map 
        component functions. This can be a polynomial, a Hermite function, a
        radial basis function, or similar.
        
        Variables:
        
            term
                [variable] : either an empty list, a list, or a string, which
                specifies what function type this term is supposed to be. 
                Empty lists correspond to a constant, lists specify polynomials
                or Hermite functions, and strings denote special terms such as
                radial basis functions.
                
            mode - [default = 'standard']
                [string] : a keyword which defines whether the term's string 
                returned should be conventional ('standard') or its derivative 
                ('derivative').
                
            k - [default = None]
                [integer or None] : an integer specifying what dimension of the
                samples the 'term' corresponds to. Used to clarify with respect
                to what variable we take the derivative.
        """
        
        import copy
        
        # First, check for input errors ---------------------------------------
        
        # Check if mode is valid
        if mode not in ['standard','derivative']:
            raise ValueError("Mode must be either 'standard' or 'derivative'. Unrecognized mode: "+str(mode))
                
        # If derivative, check if k is specified
        if mode == 'derivative' and k is None:
            raise ValueError("If mode == 'derivative', specify an integer for k to inform with regards to which variable we take the derivative. Currently specified: k = "+str(k))

        # If derivative, check if k is an integer
        if mode == 'derivative' and type(k) is not int:
            raise ValueError("If mode == 'derivative', specify an integer for k to inform with regards to which variable we take the derivative. Currently specified: k = "+str(k))

        # Initiate the modifier log -------------------------------------------

        # This variable returns information about whether there is anything 
        # special about this term. If this is not None, it is a dictionary with
        # the following possible keys:
        #   "constant"  this is a constant term
        #   "ST"        this is a RBF-based special term
        #   "HF"        this is a polynomial with a Hermite function modifier
        #   "LIN"       this is a polynomial with a linearization modifier
        #   "HFLIN"     this is a polynomial with both modifiers
        
        modifier_log    = {}


        # =====================================================================
        # Constant
        # =====================================================================
        
        # If the entry is an empty list, add a constant
        if term == []:
            
            if mode == 'standard':
                
                # Construct the string
                string  = "np.ones(__x__.shape[:-1])"
                
            elif mode == 'derivative':
                
                # Construct the string
                string  = "np.zeros(__x__.shape[:-1])"
                
            # Log this term
            modifier_log    = {"constant"   : None}
            
        # =====================================================================
        # Special term
        # =====================================================================
            
        # If the entry is a string, it denotes a special term
        elif type(term) == str:
            
            # Split the string 
            STtype,i    = term.split(' ')
            
            # Log this term
            modifier_log    = {"ST" : int(i)}
            
            # --------------------------------------------------------------W---
            # Left edge term
            # -----------------------------------------------------------------
            
            if STtype.lower() == "let": # The special term is a left edge term
            
                if mode == 'standard':
                    
                    # https://www.wolframalpha.com/input/?i=%28%28x+-+%5Cmu%29*%281-erf%28%28x+-+%5Cmu%29%2F%28sqrt%282%29*%5Csigma%29%29%29+-+%5Csigma*sqrt%282%2F%5Cpi%29*exp%28-%28%28x+-+%5Cmu%29%2F%28sqrt%282%29*%5Csigma%29%29%5E2%29%29%2F2
                    
                    # Construct the string
                    string      = "((__x__[...,"+i+"] - __mu__)*(1-scipy.special.erf((__x__[...,"+i+"] - __mu__)/(np.sqrt(2)*__scale__))) - __scale__*np.sqrt(2/np.pi)*np.exp(-((__x__[...,"+i+"] - __mu__)/(np.sqrt(2)*__scale__))**2))/2"
                    
                elif mode == 'derivative':
                    
                    # https://www.wolframalpha.com/input/?i=derivative+of+%28%28x+-+%5Cmu%29*%281-erf%28%28x+-+%5Cmu%29%2F%28sqrt%282%29*%5Csigma%29%29%29+-+%5Csigma*sqrt%282%2F%5Cpi%29*exp%28-%28%28x+-+%5Cmu%29%2F%28sqrt%282%29*%5Csigma%29%29%5E2%29%29%2F2+wrt+x
                    
                    # Construct the string
                    if int(i) == k:
                        
                        string      = "(1 - scipy.special.erf((__x__[...,"+i+"] - __mu__)/(np.sqrt(2)*__scale__)))/2"
                        
                    else:
                        
                        string      = "np.zeros(__x__.shape[:-1])"
                    
            # -----------------------------------------------------------------
            # Right edge term
            # -----------------------------------------------------------------
            
            elif STtype.lower() == "ret": # The special term is a right edge term
            
                if mode == 'standard':
                    
                    # https://www.wolframalpha.com/input/?i=%28%28x+-+%5Cmu%29*%281%2Berf%28%28x+-+%5Cmu%29%2F%28sqrt%282%29*%5Csigma%29%29%29+%2B+%5Csigma*sqrt%282%2F%5Cpi%29*exp%28-%28%28x+-+%5Cmu%29%2F%28sqrt%282%29*%5Csigma%29%29%5E2%29%29%2F2
                    
                    # Construct the string
                    string      = "((__x__[...,"+i+"] - __mu__)*(1+scipy.special.erf((__x__[...,"+i+"] - __mu__)/(np.sqrt(2)*__scale__))) + __scale__*np.sqrt(2/np.pi)*np.exp(-((__x__[...,"+i+"] - __mu__)/(np.sqrt(2)*__scale__))**2))/2"
                    
                elif mode == 'derivative':
                    
                    # https://www.wolframalpha.com/input/?i=derivative+of+%28%28x+-+%5Cmu%29*%281%2Berf%28%28x+-+%5Cmu%29%2F%28sqrt%282%29*%5Csigma%29%29%29+%2B+%5Csigma*sqrt%282%2F%5Cpi%29*exp%28-%28%28x+-+%5Cmu%29%2F%28sqrt%282%29*%5Csigma%29%29%5E2%29%29%2F2+wrt+x
                    
                    # Construct the string
                    if int(i) == k:
                    
                        string      = "(1 + scipy.special.erf((__x__[...,"+i+"] - __mu__)/(np.sqrt(2)*__scale__)))/2"
                    
                    else:
                        
                        string      = "np.zeros(__x__.shape[:-1])"
                    
            # -----------------------------------------------------------------
            # Radial basis function
            # -----------------------------------------------------------------
            
            elif STtype.lower() == "rbf": # The special term is a right edge term
            
                if mode == 'standard':
                    
                    # https://www.wolframalpha.com/input/?i=1%2F%28sqrt%282*%5Cpi%29*%5Csigma%29*exp%28-%28x+-+%5Cmu%29**2%2F%282*%5Csigma%5E2%29%29
                    
                    # Construct the string
                    string      = "np.exp(-((__x__[...,"+i+"] - __mu__)/__scale__)**2/2)/(__scale__*np.sqrt(2*np.pi))"
                    
                elif mode == 'derivative':
                    
                    # https://www.wolframalpha.com/input?i=derivative+of+exp%28-%28%28x+-+%5Cmu%29%2F%5Csigma%29**2%2F2%29%2F%28%5Csigma*sqrt%282*%5Cpi%29%29+wrt+x
                    
                    # Construct the string
                    if int(i) == k:
                        
                        string      = "-(__x__[...,"+i+"] - __mu__)/(np.sqrt(2*np.pi)*__scale__**3)*np.exp(-((__x__[...,"+i+"]-__mu__)/__scale__)**2/2)"
                        
                    else:
                        
                        string      = "np.zeros(__x__.shape[:-1])"
                    
            # -----------------------------------------------------------------
            # Integrated radial basis function
            # -----------------------------------------------------------------
            
            elif STtype.lower() == "irbf": # The special term is a right edge term
            
                if mode == 'standard':
                    
                    # https://www.wolframalpha.com/input/?i=%281+%2B+erf%28%28x+-+%5Cmu%29%2F%28sqrt%282%29*%5Csigma%29%29%29%2F2
                    
                    # Construct the string
                    string      = "(1 + scipy.special.erf((__x__[...,"+i+"] - __mu__)/(np.sqrt(2)*__scale__)))/2"
                    
    
                elif mode == 'derivative':
                    
                    # https://www.wolframalpha.com/input/?i=derivative+of+%281+%2B+erf%28%28x+-+%5Cmu%29%2F%28sqrt%282%29*%5Csigma%29%29%29%2F2+wrt+x
                    
                    # Construct the string
                    if int(i) == k:
                        
                        string      = "1/(np.sqrt(2*np.pi)*__scale__)*np.exp(-(__x__[...,"+i+"] - __mu__)**2/(2*__scale__**2))"
                    
                    else:
                        
                        string      = "np.zeros(__x__.shape[:-1])"
                    
            # -----------------------------------------------------------------
            # Unrecognized special term
            # -----------------------------------------------------------------
                
            else:
                
                raise ValueError("Special term '"+str(STtype)+"' not "+\
                    "understood. Currently, only LET, RET, iRBF, and RBF "+\
                    "are implemented.")
            
            
        # =====================================================================
        # Polynomial term
        # =====================================================================
        
        # Otherwise, it is a standard polynomial term
        else:
            
            # -----------------------------------------------------------------
            # Check for modifiers
            # -----------------------------------------------------------------
            
            # Check for Hermite function modifier       
            # TRUE if modifier is active, else FALSE
            hermite_function_modifier = np.asarray(
                [True if i == 'HF' else False for i in term],
                dtype=bool).any()
            
            # Check for linearization modifier  
            # TRUE if modifier is active, else FALSE                  
            linearize = np.asarray(
                [True if i == 'LIN' else False for i in term],
                dtype=bool).any()
            
            # Check if linearization is also activated
            if linearize and self.linearization is None:
                raise Exception("'LIN' modifier specified in variable monotone, but the variable linearization is defined as None. Please specify a scalar linearization or remove the 'LIN' modifier.")
            
            # Remove all string-based modifiers
            term    = [i for i in term if type(i) != str]
            
            # -----------------------------------------------------------------
            # Construct the polynomial term
            # -----------------------------------------------------------------
            
            # Extract the unique entries and their counts
            ui,ct   = np.unique(term, return_counts = True)
            
            
            # Both Hermite function and linearization modifiers are active
            if hermite_function_modifier and linearize: 
            
                # Log this term
                modifier_log    = {
                    "HFLIN"     : None}
            
            # Hermite function modifiers is active
            elif hermite_function_modifier:
                
                # Log this term
                modifier_log    = {
                    "HF"        : None}
            
            # Linearization modifiers is active
            elif linearize:
            
                # Log this term
                modifier_log    = {
                    "LIN"       : None}
            
            # Add a "variables" key to the modifier_log, if it does not already exist
            if "variables" not in list(modifier_log.keys()):
                modifier_log["variables"]   = {}

            # Create an empty string
            string  = ""
            
            # Go through all unique entries
            for i in range(len(ui)):
                
                # Create an array of polynomial coefficients
                dummy_coefficients      = [0.]*ct[i] + [1.]
                
                # Normalize the influence of Hermite functions
                if hermite_function_modifier:
                    
                    # Evaluate a naive Hermite function
                    hf_x    = np.linspace(-100,100,100001)
                    hfeval  = self.polyfunc(dummy_coefficients)(hf_x)*np.exp(-hf_x**2/4)
                    
                    # Scale the polynomial coefficient to normalize its maximum value
                    dummy_coefficients[-1]  = 1/np.max(np.abs(hfeval))
                    
                # -------------------------------------------------------------
                # Standard polynomial
                # -------------------------------------------------------------
                                
                if mode == 'standard' or (mode == 'derivative' and ui[i] != k):
                    
                    # Create a variable key
                    key     = "P_"+str(ui[i])+"_O_"+str(ct[i])
                    if hermite_function_modifier:
                        key     += "_HF"
                    if linearize:
                        key     += "_LIN"
                    
                    # Set up function -----------------------------------------
                    
                    # Extract the polynomial 
                    var         = copy.copy(self.polyfunc_str)
                    
                    # Open outer paranthesis
                    var         += "(["
                    
                    # Add polynomial coefficients
                    for dc in dummy_coefficients:
                        var     += str(dc)+","
                    
                    # Remove the last ","
                    var         = var[:-1]
                    
                    # Close outer paranthesis
                    var         += "])"
                    
                    # Add variable --------------------------------------------
                    
                    var         += "(__x__[...,"+str(ui[i])+"])"
                    
                    # Add Hermite function ------------------------------------
                    
                    if hermite_function_modifier:
                    
                        var         += "*np.exp(-__x__[...,"+str(ui[i])+"]**2/4)"
                        
                    # Save the variable ---------------------------------------
                    if key not in list(modifier_log["variables"].keys()):
                        modifier_log["variables"][key]  = copy.copy(var)
                    
                    # Add the variable to the string --------------------------
                    string      += copy.copy(key)
                        
                    # Add a multiplier, in case there are more terms 
                    string      += " * "
                    
                # -------------------------------------------------------------
                # Derivative of polynomial
                # -------------------------------------------------------------
    
                elif mode == 'derivative':
                    
                    # Create a variable key for the standard polynomial
                    key     = "P_"+str(ui[i])+"_O_"+str(ct[i])
                    
                    # Create a variable key for its derivative
                    keyder  = "P_"+str(ui[i])+"_O_"+str(ct[i])+"_DER"
                    
                    
                    # Set up function -----------------------------------------
                    
                    # Find the derivative coefficients
                    dummy_coefficients_der  = self.polyfunc_der(dummy_coefficients)
                    
                    # Extract the polynomial 
                    varder      = copy.copy(self.polyfunc_str)
                    
                    # Open outer paranthesis
                    varder      += "(["
                    
                    # Add polynomial coefficients
                    for dc in dummy_coefficients_der:
                        varder  += str(dc)+","
                    
                    # Remove the last ","
                    varder      = varder[:-1]
                    
                    # Close outer paranthesis
                    varder      += "])"
                    
                    # Add variable --------------------------------------------
                
                    varder      += "(__x__[...,"+str(ui[i])+"])"
                    
                    # Save the variable ---------------------------------------
                    if keyder not in list(modifier_log["variables"].keys()):
                        modifier_log["variables"][keyder]   = copy.copy(varder)
                    
                    # Add the variable to the string --------------------------
                    if not hermite_function_modifier:
                        string      += copy.copy(varder)
                    
                    # Add Hermite function ------------------------------------
                    
                    # https://www.wolframalpha.com/input/?i=derivative+of+f%28x%29*exp%28-x%5E2%2F4%29+wrt+x
                    
                    if hermite_function_modifier:
                        
                        # If we have a hermite function modifier, we also need
                        # the original form of the polynomial
                        
                        # Set up function -------------------------------------
                        
                        # Extract the polynomial 
                        varbase     = copy.copy(self.polyfunc_str)
                        
                        # Open outer paranthesis
                        varbase     += "(["
                        
                        # Add polynomial coefficients
                        for dc in dummy_coefficients:
                            varbase     += str(dc)+","
                        
                        # Remove the last ","
                        varbase     = varbase[:-1]
                        
                        # Close outer paranthesis
                        varbase     += "])"
                        
                        # Add variable ----------------------------------------
                    
                        varbase     += "(__x__[...,"+str(ui[i])+"])"
                        
                        # Save the variable -----------------------------------
                        if key not in list(modifier_log["variables"].keys()):
                            modifier_log["variables"][key]   = copy.copy(varbase)
                        
                        # Now we can construct the actual derivative ----------
                        
                        string      = "-1/2*np.exp(-__x__[...,"+str(ui[i])+"]**2/4)*(__x__[...,"+str(ui[i])+"]*"+key+" - 2*"+keyder+")"
                    
                    # Add a multiplier, in case there are more terms ----------
                    string      += " * "
            
            # Remove the last multiplier " * "
            string      = string[:-3]
            
            # If the variable we take the derivative against is not in the term,
            # overwrite the string with zeros
            if mode == 'derivative' and k not in ui:
                
                # Overwrite string with zeros
                string      = "np.zeros(__x__.shape[:-1])"
                
                
        return string, modifier_log
        
    

    def function_constructor_alternative(self, k = None):
        
        """
        This function assembles the string for the monotone and nonmonotone map
        components, then converts these strings into functions.
        
        Variables:
        
            k - [default = None]
                [integer or None] : an integer specifying what dimension of the
                samples the 'term' corresponds to. Used to clarify with respect
                to what dimension we build this basis function
        """
        
        import numpy as np
        import copy
        
        if k is None:
            
            # Do we only construct the functions for one dimension?
            partial_construction    = False
            
            # Construct the functions for all dimensions
            Ks                      = np.arange(self.D)
            
            # Initialize empty lists for the monotone part functions, their 
            # corresponding strings, and coefficients.
            self.fun_mon            = []
            self.fun_mon_strings    = []
            self.coeffs_mon         = []
            
            # Initialize empty lists for the nonmonotone part functions, their 
            # corresponding strings, and coefficients.
            self.fun_nonmon         = []
            self.fun_nonmon_strings = []
            self.coeffs_nonmon      = []
            
            # Check for any special terms
            self.check_for_special_terms()
            self.determine_special_term_locations()
            
        elif np.isscalar(k):
            
            # Do we only construct the functions for one dimension?
            partial_construction    = True
            
            # Construct the functions only for this dimension
            Ks                      = [k]
            
        else:
            
            # Input is not recognized. Raise an error.
            raise Exception("'k' for function_constructor_alternative must be either None or an integer.")
        
        # Go through all terms
        for k in Ks:
            
            # =================================================================
            # =================================================================
            # Step 1: Build the monotone function
            # =================================================================
            # =================================================================
            
            # Define modules to load
            modules = ["import numpy as np","import copy"]
                    
            # =================================================================
            # Extract the terms
            # =================================================================  
                
            # Define the terms composing the transport map component
            terms   = []
            
            # Prepare a counter for the special terms
            ST_counter  = np.zeros(self.X.shape[-1],dtype=int)
            
            # Prepare a dictionary for precalculated variables
            dict_precalc    = {}
            
            # Mark which of these are special terms, in case we want to create
            # permutations of multiple RBFS
            ST_indices      = []
            
            # Go through all terms
            for i,entry in enumerate(self.monotone[k]):
                
                # -------------------------------------------------------------
                # Convert the map specification to a function
                # -------------------------------------------------------------
                
                # Find the term's function
                term, modifier_log = self.write_basis_function(
                    term    = entry,
                    mode    = 'standard')

                # -------------------------------------------------------------
                # Extract any precalculations, where applicable
                # -------------------------------------------------------------
                
                # If this term includes and precalculations, extract them
                if "variables" in list(modifier_log.keys()):
                    
                    # There are precalculating variables. Go through each
                    for key in list(modifier_log["variables"].keys()):
                        
                        # Have we logged this one already?
                        if key not in list(dict_precalc.keys()):
                            
                            # No, we haven't. Add it.
                            dict_precalc[key]   = copy.copy(modifier_log["variables"][key]).replace("__x__","x")
                            
                            # Wait a moment! Are we linearizing this term?
                            if key.endswith("_LIN"):
                                
                                # Yes, we are! What dimension is this?
                                d   = int(copy.copy(key).split("_")[1])
                                
                                # Edit the term
                                dict_precalc[key]   = \
                                    copy.copy(dict_precalc[key]).replace("__x__","x_trc") + " * " + \
                                    "(1 - vec[:,"+str(d)+"]/"+str(self.linearization_increment)+") + " + \
                                    copy.copy(dict_precalc[key]).replace("__x__","x_ext") + " * " + \
                                    "vec[:,"+str(d)+"]/"+str(self.linearization_increment)
                            
                # -------------------------------------------------------------
                # Post-processing for special terms
                # -------------------------------------------------------------
                
                # Is this term a special term?
                if "ST" in list(modifier_log.keys()):
                    
                    # Mark this term as a special one
                    ST_indices.append(i)
                
                    # Yes, it is. Add additional modules to load, if necessary
                    if "import scipy.special" not in modules:
                        modules     .append("import scipy.special")
                        
                    # Extract this special term's dimension
                    idx     = modifier_log["ST"]
                
                    # Is this a cross-term? 
                    # Cross-terms are stored in a separate key; access it, if
                    # necessary.
                    if k+self.skip_dimensions != idx:
                        # Yes, it is.
                        ctkey   = "['cross-terms']"
                    else:
                        # No, it isn't.
                        ctkey   = ""
                
                    # Replace __mu__ with the correct ST location variable
                    term    = term.replace(
                        "__mu__",
                        "self.special_terms["+str(k+self.skip_dimensions)+"]"+ctkey+"["+str(idx)+"]['centers']["+str(ST_counter[idx])+"]")
                    
                    # Replace __scale__ with the correct ST location variable
                    # self.special_terms[k][d]
                    
                    term    = term.replace(
                        "__scale__",
                        "self.special_terms["+str(k+self.skip_dimensions)+"]"+ctkey+"["+str(idx)+"]['scales']["+str(ST_counter[idx])+"]")
                    
                    # Increment the special term counter
                    ST_counter[idx]  += 1
                    
                # -------------------------------------------------------------
                # Add the term to the list
                # -------------------------------------------------------------
                
                # If any dummy __x__ remain, replace them
                term    = term.replace("__x__","x")
                
                # Store the term
                terms   .append(copy.copy(term))
                
            # -----------------------------------------------------------------
            # If there are special cross-terms, create them
            # -----------------------------------------------------------------
            
            # Are there multiple special terms?
            # if np.sum([True if x != k else False for x in list(self.special_terms[k+self.skip_dimensions].keys())]) > 1:
            # if np.sum([True if x != 0 else False for x in self.RBF_counter_m[k,:]]) > 1:
            if 'cross-terms' in list(self.special_terms[k+self.skip_dimensions].keys()):
                
                import itertools
                
                # Yes, there are multiple special terms. Extract these terms.
                RBF_terms   = [terms[i] for i in ST_indices]
                
                # Check what variables these terms are affiliated with
                RBF_terms_dim   = - np.ones(len(RBF_terms),dtype=int)
                for ki in range(k+1+self.skip_dimensions):
                    for i,term in enumerate(RBF_terms):
                        if "x[...,"+str(ki)+"]" in term:
                            RBF_terms_dim[i]    = ki
                RBF_terms_dims  = np.unique(np.asarray(RBF_terms_dim))
                            
                # Create a dictionary with the different terms
                RBF_terms_dict  = {}
                for i in RBF_terms_dims:
                    RBF_terms_dict[i]   = [RBF_terms[j] for j in range(len(RBF_terms)) if RBF_terms_dim[j] == i]
                    
                # Create all combinations of terms
                RBF_terms_grid  = copy.deepcopy(RBF_terms_dict[RBF_terms_dims[0]])
                for i in RBF_terms_dims[1:]:
                    
                    # Create a grid with the next dimension
                    RBF_terms_grid  = list(itertools.product(
                        RBF_terms_grid,
                        copy.deepcopy(RBF_terms_dict[i])))
                    
                    # Convert this list of tuples into a new list of strings
                    RBF_terms_grid  = \
                        [entry[0]+"*"+entry[1] for entry in RBF_terms_grid]
                        
                # Now remove all original RBF terms
                terms   = [entry for i,entry in enumerate(terms) if i not in ST_indices]
                
                # Now add all the grid terms
                terms   += RBF_terms_grid
            
            # -----------------------------------------------------------------
            # Add monotone coefficients
            # -----------------------------------------------------------------
            
            if not partial_construction:
                # Append the parameters
                self.coeffs_mon     .append(np.ones(len(terms))*self.coeffs_init)
                
            # =================================================================
            # Assemble the monotone function
            # =================================================================
            
            # Prepare the basis string
            string  = "def fun(x,self):\n\t\n\t"
            
            # -----------------------------------------------------------------
            # Load module requirements
            # -----------------------------------------------------------------
            
            for entry in modules:
                string  += copy.copy(entry)+"\n\t"
            string  += "\n\t" # Another line break for legibility
            
            # -----------------------------------------------------------------
            # Prepare linearization, if necessary
            # -----------------------------------------------------------------
            
            # If linearization is active, truncate the input x
            if self.linearization is not None:
                
                # First, find our which parts are outside the linearization hypercube
                string  += "vec_below = copy.copy(x) - self.linearization_threshold[:,0][np.newaxis,:];\n\t"
                string  += "vec_below[vec_below >= 0] = 0;\n\t" # Set all values above to zero
                string  += "vec_above = copy.copy(x) - self.linearization_threshold[:,1][np.newaxis,:];\n\t"
                string  += "vec_above[vec_above <= 0] = 0;\n\t" # Set all values below to zero
                string  += "vec = vec_above + vec_below;\n\t"
                
                # Then convert the two arrays to boolean markers
                string  += "below = (vec_below < 0);\n\t" # Find all particles BELOW the lower linearization band
                string  += "above = (vec_above > 0);\n\t" # Find all particles ABOVE the upper linearization band
                string  += "shift = np.asarray(below,dtype=float) + np.asarray(above,dtype=float);\n\t" # This is a matrix where all entries outside the linearization bands are 1 and all entries inside are 0
                
                # Truncate all values outside the hypercube
                string  += "x_trc = copy.copy(x);\n\t"
                string  += "for d in range(x.shape[1]):\n\t\t"
                string  += "x_trc[below[:,d],d] = self.linearization_threshold[d,0];\n\t\t" # All values below the linearization band of this dimension are snapped to its border
                string  += "x_trc[above[:,d],d] = self.linearization_threshold[d,1];\n\t"   # All values above the linearization band of this dimension are snapped to its border
                
                # Add a space to the next block
                string  += "\n\t"
                
                # Also crate an extrapolated version of x_trc
                string  += "x_ext = copy.copy(x_trc);\n\t"
                string  += "x_ext += shift*"+str(self.linearization_increment)+";\n\t" # Offset all values which have been snapped by a small increment
                
                # Add a space to the next block
                string  += "\n\t"
                
            # -----------------------------------------------------------------
            # Prepare precalculated variables
            # -----------------------------------------------------------------
            
            # Add all precalculation terms
            for key in list(dict_precalc.keys()):
                
                string  += key + " = " + copy.copy(dict_precalc[key]) + ";\n\t"
                
            # -----------------------------------------------------------------
            # Assemble function output
            # -----------------------------------------------------------------
                
            # Prepare the result string
            if len(terms) == 1: # Only a single term, no need for stacking
                
                string  += "result = "+copy.copy(terms[0])+"[:,np.newaxis];\n\t\n\t"
                
            else: # If we have more than one term, start stacking the result
                
                # Prepare the stack
                string  += "result = np.stack((\n\t\t"
    
                # Go through each entry in terms, add them one by one
                for entry in terms:
                    string  += copy.copy(entry) + ",\n\t\t"
                    
                # Remove the last ",\n\t\t" and close the stack
                string  = string[:-4]
                string  += "),axis=-1)\n\t\n\t"
                
            # Return the result
            string  += "return result"
            
            # -----------------------------------------------------------------
            # Finish function construction
            # -----------------------------------------------------------------
    
            if not partial_construction:
    
                # Append the function string
                self.fun_mon_strings    .append(string)
    
                # Create an actual function
                funstring   = "fun_mon_"+str(k)
                exec(string.replace("fun",funstring), globals())
                exec("self.fun_mon.append(copy.deepcopy("+funstring+"))")
                
            else:
                
                # Insert the function string
                self.fun_mon_strings[k]  = copy.copy(string)

                # Create an actual function
                funstring   = "fun_nonmon_"+str(k)
                exec(string.replace("fun",funstring), globals())
                exec("self.fun_mon[k] = copy.deepcopy("+funstring+")")
                
            # =================================================================
            # =================================================================
            # Step 2: Build the nonmonotone function
            # =================================================================
            # =================================================================
            
            if not partial_construction:
                
                # Append the parameters
                self.coeffs_nonmon  .append(np.ones(len(self.nonmonotone[k]))*self.coeffs_init)
            
            # Define modules to load
            modules = ["import numpy as np","import copy"]
                
            # =================================================================
            # Extract the terms
            # =================================================================  
                
            # Define the terms composing the transport map component
            terms   = []
            
            # Prepare a counter for the special terms
            ST_counter  = np.zeros(self.X.shape[-1],dtype=int)
            
            # Prepare a dictionary for precalculated variables
            dict_precalc    = {}
            
            # Go through all terms
            for entry in self.nonmonotone[k]:
                
                # -------------------------------------------------------------
                # Convert the map specification to a function
                # -------------------------------------------------------------
                
                # Find the term's function
                term, modifier_log = self.write_basis_function(
                    term    = entry,
                    mode    = 'standard')
                
                # -------------------------------------------------------------
                # Extract any precalculations, where applicable
                # -------------------------------------------------------------
                
                # If this term includes and precalculations, extract them
                if "variables" in list(modifier_log.keys()):
                    
                    # There are precalculating variables. Go through each
                    for key in list(modifier_log["variables"].keys()):
                        
                        # Have we logged this one already?
                        if key not in list(dict_precalc.keys()):
                            
                            # No, we haven't. Add it.
                            dict_precalc[key]   = copy.copy(modifier_log["variables"][key]).replace("__x__","x")
                            
                            # Wait a moment! Are we linearizing this term?
                            if key.endswith("_LIN"):
                                
                                # Yes, we are! What dimension is this?
                                d   = int(copy.copy(key).split("_")[1])
                                
                                # Edit the term
                                dict_precalc[key]   = \
                                    copy.copy(dict_precalc[key]).replace("__x__","x_trc") + " * " + \
                                    "(1 - vec[:,"+str(d)+"]/"+str(self.linearization_increment)+") + " + \
                                    copy.copy(dict_precalc[key]).replace("__x__","x_ext") + " * " + \
                                    "vec[:,"+str(d)+"]/"+str(self.linearization_increment)
                            
                # -------------------------------------------------------------
                # Post-processing for special terms
                # -------------------------------------------------------------
                
                # Is this term a special term?
                if "ST" in list(modifier_log.keys()):
                
                    # Yes, it is. Add additional modules to load, if necessary
                    if "import scipy.special" not in modules:
                        modules     .append("import scipy.special")
                        
                    # Extract this special term's dimension
                    idx     = modifier_log["ST"]
                
                    # Replace __mu__ with the correct ST location variable
                    term    = term.replace(
                        "__mu__",
                        "self.special_terms["+str(k+self.skip_dimensions)+"]["+str(idx)+"]['centers']["+str(ST_counter[idx])+"]")
                    
                    # Replace __scale__ with the correct ST location variable
                    term    = term.replace(
                        "__scale__",
                        "self.special_terms["+str(k+self.skip_dimensions)+"]["+str(idx)+"]['scales']["+str(ST_counter[idx])+"]")
                    
                    # Increment the special term counter
                    ST_counter[idx]  += 1
                    
                # -------------------------------------------------------------
                # Add the term to the list
                # -------------------------------------------------------------
                
                # If any dummy __x__ remain, replace them
                term    = term.replace("__x__","x")
                
                # Store the term
                terms   .append(copy.copy(term))
                
            # =================================================================
            # Assemble the monotone function
            # =================================================================
            
            # Only assemble the function if there actually is a nonmonotone term
            if len(self.nonmonotone[k]) > 0:
            
                # Prepare the basis string
                string  = "def fun(x,self):\n\t\n\t"
                
                # -------------------------------------------------------------
                # Load module requirements
                # -------------------------------------------------------------
                
                for entry in modules:
                    string  += copy.copy(entry)+"\n\t"
                string  += "\n\t" # Another line break for legibility
                
                # -------------------------------------------------------------
                # Prepare linearization, if necessary
                # -------------------------------------------------------------
                
                # If linearization is active, truncate the input x
                if self.linearization is not None:
                    
                    # First, find our which parts are outside the linearization hypercube
                    string  += "vec_below = copy.copy(x) - self.linearization_threshold[:,0][np.newaxis,:];\n\t"
                    string  += "vec_below[vec_below >= 0] = 0;\n\t" # Set all values above to zero
                    string  += "vec_above = copy.copy(x) - self.linearization_threshold[:,1][np.newaxis,:];\n\t"
                    string  += "vec_above[vec_above <= 0] = 0;\n\t" # Set all values below to zero
                    string  += "vec = vec_above + vec_below;\n\t"
                    
                    # Then convert the two arrays to boolean markers
                    string  += "below = (vec_below < 0);\n\t" # Find all particles BELOW the lower linearization band
                    string  += "above = (vec_above > 0);\n\t" # Find all particles ABOVE the upper linearization band
                    string  += "shift = np.asarray(below,dtype=float) + np.asarray(above,dtype=float);\n\t" # This is a matrix where all entries outside the linearization bands are 1 and all entries inside are 0
                    
                    # Truncate all values outside the hypercube
                    string  += "x_trc = copy.copy(x);\n\t"
                    string  += "for d in range(x.shape[1]):\n\t\t"
                    string  += "x_trc[below[:,d],d] = self.linearization_threshold[d,0];\n\t\t" # All values below the linearization band of this dimension are snapped to its border
                    string  += "x_trc[above[:,d],d] = self.linearization_threshold[d,1];\n\t"   # All values above the linearization band of this dimension are snapped to its border
                    
                    # Add a space to the next block
                    string  += "\n\t"
                    
                    # Also crate an extrapolated version of x_trc
                    string  += "x_ext = copy.copy(x_trc);\n\t"
                    string  += "x_ext += shift*"+str(self.linearization_increment)+";\n\t" # Offset all values which have been snapped by a small increment
                    
                    # Add a space to the next block
                    string  += "\n\t"
                    
                # -------------------------------------------------------------
                # Prepare precalculated variables
                # -------------------------------------------------------------
                
                # Add all precalculation terms
                for key in list(dict_precalc.keys()):
                    
                    string  += key + " = " + copy.copy(dict_precalc[key]) + ";\n\t"
                    
                # -------------------------------------------------------------
                # Assemble function output
                # -------------------------------------------------------------
                    
                # Prepare the result string
                if len(terms) == 1: # Only a single term, no need for stacking
                    
                    string  += "result = "+copy.copy(terms[0])+"[:,np.newaxis];\n\t\n\t"
                    
                else: # If we have more than one term, start stacking the result
                    
                    # Prepare the stack
                    string  += "result = np.stack((\n\t\t"
        
                    # Go through each entry in terms, add them one by one
                    for entry in terms:
                        string  += copy.copy(entry) + ",\n\t\t"
                        
                    # Remove the last ",\n\t\t" and close the stack
                    string  = string[:-4]
                    string  += "),axis=-1)\n\t\n\t"
                    
                # Return the result
                string  += "return result"
                
                # -------------------------------------------------------------
                # Finish function construction
                # -------------------------------------------------------------
        
                if not partial_construction:
        
                    # Append the function string
                    self.fun_nonmon_strings    .append(string)
                    
                    # Create an actual function
                    funstring   = "fun_nonmon_"+str(k)
                    exec(string.replace("fun",funstring), globals())
                    exec("self.fun_nonmon.append(copy.deepcopy("+funstring+"))")
                    
                else:
                    
                    # Insert the function string
                    self.fun_nonmon_strings[k]  = copy.copy(string)
    
                    # Create an actual function
                    funstring   = "fun_nonmon_"+str(k)
                    exec(string.replace("fun",funstring), globals())
                    exec("self.fun_nonmon[k] = copy.deepcopy("+funstring+")")
                
            else: # There are NO non-monotone terms
            
                # Create a function which just returns None
                string      = "def fun(x,self):\n\t"
                string      += "return None"
                
                if not partial_construction:
        
                    # Append the function string
                    self.fun_nonmon_strings     .append(string)
                    
                    # Create an actual function
                    funstring   = "fun_nonmon_"+str(k)
                    exec(string.replace("fun",funstring), globals())
                    exec("self.fun_nonmon.append(copy.deepcopy("+funstring+"))")
                    
                else:
                    
                    # Insert the function string
                    self.fun_nonmon_strings[k]  = copy.copy(string)
    
                    # Create an actual function
                    funstring   = "fun_nonmon_"+str(k)
                    exec(string.replace("fun",funstring), globals())
                    exec("self.fun_nonmon[k] = copy.deepcopy("+funstring+")")
                
        # =================================================================
        # =================================================================
        # Step 3: Finalize
        # =================================================================
        # =================================================================

        # If monotonicity mode is 'separable monotonicity', we also require the
        # derivative of the monotone part of the map
        if self.monotonicity.lower() == "separable monotonicity":
                
            self.function_derivative_constructor_alternative()
                
                
        return
    
    

    def function_derivative_constructor_alternative(self):
        
        """
        This function is the complement to 'function_constructor_alternative',
        but instead constructs the derivative of the map's component functions.
        It constructs the functions' strings, then converts them into 
        functions.
        """
    
        import numpy as np
        import copy
        
        self.der_fun_mon            = []
        self.der_fun_mon_strings    = []
        
        self.optimization_constraints_lb = []
        self.optimization_constraints_ub = []
        
        # Find out how many function terms we are building
        K       = len(self.monotone)
        
        # Go through all terms
        for k in range(K):
            
            # =================================================================
            # =================================================================
            # Step 1: Build the monotone function
            # =================================================================
            # =================================================================
            
            # Set optimization constraints
            self.optimization_constraints_lb.append(np.zeros(len(self.monotone[k])))
            self.optimization_constraints_ub.append(np.ones(len(self.monotone[k]))*np.inf)
            
            # Define modules to load
            modules = ["import numpy as np", "import copy"]
            
            # Define the terms composing the transport map component
            terms   = []
            
            # Prepare a counter for the special terms
            ST_counter  = np.zeros(self.X.shape[-1],dtype=int)
            
            # Mark which of these are special terms, in case we want to create
            # permutations of multiple RBFS
            ST_indices      = []
            
            # Go through all terms, extract terms for precalculation
            dict_precalc    = {}
            for j,entry in enumerate(self.monotone[k]):
                
                # -------------------------------------------------------------
                # Convert the map specification to a function
                # -------------------------------------------------------------
                
                # Find the term's function
                term, modifier_log = self.write_basis_function(
                    term    = entry,
                    mode    = 'derivative',
                    k       = k + self.skip_dimensions)
                
                # -------------------------------------------------------------
                # If this is a constant term, undo the lower constraint
                # -------------------------------------------------------------
                
                if "constant" in list(modifier_log.keys()):
                
                    # Assign linear constraints
                    self.optimization_constraints_lb[k][j]  = -np.inf
                    self.optimization_constraints_ub[k][j]  = +np.inf
                
                # -------------------------------------------------------------
                # Extract any precalculations, where applicable
                # -------------------------------------------------------------
                
                # If this term includes and precalculations, extract them
                if "variables" in list(modifier_log.keys()):
                    
                    # There are precalculating variables. Go through each
                    for key in list(modifier_log["variables"].keys()):
                        
                        # Have we logged this one already?
                        if key not in list(dict_precalc.keys()):
                            
                            # No, we haven't. Add it.
                            dict_precalc[key]   = copy.copy(modifier_log["variables"][key]).replace("__x__","x")
                            
                # -------------------------------------------------------------
                # Post-processing for special terms
                # -------------------------------------------------------------
                
                # Is this term a special term?
                if "ST" in list(modifier_log.keys()):
                    
                    # Mark this term as a special one
                    ST_indices.append(j)
                
                    # Yes, it is. Add additional modules to load, if necessary
                    if "import scipy.special" not in modules:
                        modules     .append("import scipy.special")
                        
                    # Extract this special term's dimension
                    idx     = modifier_log["ST"]
                    
                    # Is this a cross-term? 
                    # Cross-terms are stored in a separate key; access it, if
                    # necessary.
                    if k+self.skip_dimensions != idx:
                        # Yes, it is.
                        ctkey   = "['cross-terms']"
                    else:
                        # No, it isn't.
                        ctkey   = ""
                
                    # Replace __mu__ with the correct ST location variable
                    term    = term.replace(
                        "__mu__",
                        "self.special_terms["+str(k+self.skip_dimensions)+"]"+ctkey+"["+str(idx)+"]['centers']["+str(ST_counter[idx])+"]")
                    
                    # Replace __scale__ with the correct ST location variable
                    term    = term.replace(
                        "__scale__",
                        "self.special_terms["+str(k+self.skip_dimensions)+"]"+ctkey+"["+str(idx)+"]['scales']["+str(ST_counter[idx])+"]")
                    
                    # Increment the special term counter
                    ST_counter[idx]  += 1
                    
                # -------------------------------------------------------------
                # Add the term to the list
                # -------------------------------------------------------------
                
                # If any dummy __x__ remain, replace them
                term    = term.replace("__x__","x")
                
                # Store the term
                terms   .append(copy.copy(term))
                
            # Are there multiple special terms?
            # if np.sum([True if x != 0 else False for x in self.RBF_counter_m[k,:]]) > 1:
            # if np.sum([True if x != k else False for x in list(self.special_terms[k].keys())]) > 1:
            if 'cross-terms' in list(self.special_terms[k+self.skip_dimensions].keys()):
                
                import itertools
                
                # Yes, there are multiple special terms. Extract these terms.
                RBF_terms   = [terms[i] for i in ST_indices]
                
                # Check what variables these terms are affiliated with
                RBF_terms_dim   = - np.ones(len(RBF_terms),dtype=int)
                for ki in range(k+1+self.skip_dimensions):
                    for i,term in enumerate(RBF_terms):
                        if "x[...,"+str(ki)+"]" in term:
                            RBF_terms_dim[i]    = ki
                RBF_terms_dims  = np.unique(np.asarray(RBF_terms_dim))
                            
                # Create a dictionary with the different terms
                RBF_terms_dict  = {}
                for i in RBF_terms_dims:
                    RBF_terms_dict[i]   = [RBF_terms[j] for j in range(len(RBF_terms)) if RBF_terms_dim[j] == i]
                    
                # Create all combinations of terms
                RBF_terms_grid  = copy.deepcopy(RBF_terms_dict[RBF_terms_dims[0]])
                for i in RBF_terms_dims[1:]:
                    
                    # Create a grid with the next dimension
                    RBF_terms_grid  = list(itertools.product(
                        RBF_terms_grid,
                        copy.deepcopy(RBF_terms_dict[i])))
                    
                    # Convert this list of tuples into a new list of strings
                    RBF_terms_grid  = \
                        [entry[0]+"*"+entry[1] for entry in RBF_terms_grid]
                        
                # Now remove all original RBF terms
                terms   = [entry for i,entry in enumerate(terms) if i not in ST_indices]
                
                # Now add all the grid terms
                terms   += RBF_terms_grid
                
                
                

            # =================================================================
            # Assemble the monotone derivative function
            # =================================================================
            
            # Prepare the basis string
            string  = "def fun(x,self):\n\t\n\t"
            
            # -----------------------------------------------------------------
            # Load module requirements
            # -----------------------------------------------------------------
            
            # Add all module requirements
            for entry in modules:
                string  += copy.copy(entry)+"\n\t"
            string  += "\n\t" # Another line break for legibility
            
            # -----------------------------------------------------------------
            # Prepare linearization, if necessary
            # -----------------------------------------------------------------
            
            # If linearization is active, truncate the input x
            if self.linearization is not None:
                
                # First, find our which parts are outside the linearization hypercube
                string  += "vec_below = self.linearization_threshold[:,0][np.newaxis,:] - x;\n\t"
                string  += "vec_below[vec_below > 0] = 0;\n\t" # Set all values above to zero
                string  += "vec_above = x - self.linearization_threshold[:,1][np.newaxis,:];\n\t"
                string  += "vec_above[vec_above > 0] = 0;\n\t" # Set all values below to zero
                string  += "vec = vec_above + vec_below;\n\t"
                
                # Then convert the two arrays to boolean markers
                string  += "below = (vec_below < 0)\n\t"
                string  += "above = (vec_above > 0);\n\t"
                string  += "vecnorm = np.asarray(below,dtype=float) + np.asarray(above,dtype=float);\n\t"
                
                # Truncate all values outside the hypercube
                string  += "for d in range(x.shape[1]):\n\t\t"
                string  += "x[below[:,d],d] = self.linearization_threshold[d,0];\n\t\t"
                string  += "x[above[:,d],d] = self.linearization_threshold[d,1];\n\t"
                
                # Add a space to the next block
                string  += "\n\t"
                
                # The derivative of a linearized function outside its range is
                # constant, so we do not require x_ext
                
            # -----------------------------------------------------------------
            # Prepare precalculated variables
            # -----------------------------------------------------------------
            
            # Add all precalculation terms
            for key in list(dict_precalc.keys()):
                
                string  += key + " = " + copy.copy(dict_precalc[key]) + ";\n\t"

            # -----------------------------------------------------------------
            # Assemble function output
            # -----------------------------------------------------------------
            
            # Prepare the result string
            if len(terms) == 1: # Only a single term, no need for stacking
                
                string  += "result = "+copy.copy(terms[0])+"[:,np.newaxis];\n\t\n\t"
                
            else: # If we have more than one term, start stacking the result
                
                # Prepare the stack
                string  += "result = np.stack((\n\t\t"
    
                # Go through each entry in terms, add them one by one
                for entry in terms:
                    string  += copy.copy(entry) + ",\n\t\t"
                    
                # Remove the last ",\n\t\t" and close the stack
                string  = string[:-4]
                string  += "),axis=-1)\n\t\n\t"
                
            # Return the result
            string  += "return result"
            
            # -----------------------------------------------------------------
            # Finish function construction
            # -----------------------------------------------------------------
    
            # Append the function string
            self.der_fun_mon_strings    .append(string)

            # Create an actual function
            funstring   = "der_fun_mon_"+str(k)
            exec(string.replace("fun",funstring), globals())
            exec("self.der_fun_mon.append(copy.deepcopy("+funstring+"))")
                
        return
    
    
    def check_for_special_terms(self):
        
        """
        This function scans through the user-provided map specifications and
        seeks if there are any special terms ('RBF', 'iRBF', 'LET', 'RET') 
        among the terms of the map components. If there are, it determines
        how many there are, and informs the rest of the function where these
        special terms should be located.
        """
        
        # Number of RBFs
        self.special_terms  = {}
        
        # Go through all map components
        for k in range(self.D):
            
            # Add a key for this term
            self.special_terms[k+self.skip_dimensions]   = {}
            
            # Check all nonmonotone terms of this map component
            for entry in self.nonmonotone[k]:
                
                # If this term is a string, it denotes a special term
                if type(entry) == str:
                    
                    # Split the entry and extract its dimensional entry
                    index   = int(entry.split(' ')[1])
                    
                    # If this key does not yet exist, create it
                    if index not in list(self.special_terms[k+self.skip_dimensions].keys()):
                        self.special_terms[k+self.skip_dimensions][index]    = {
                            'counter'   : 0,
                            'centers'   : np.asarray([]),
                            'scales'    : np.asarray([])}
                    
                    # Mark it in memory
                    self.special_terms[k+self.skip_dimensions][index]['counter'] += 1
                    
            # Check all monotone terms of this map component
            for entry in self.monotone[k]:
                
                # If this term is a string, it denotes a special term
                if type(entry) == str:
                    
                    # Split the entry and extract its dimensional entry
                    index   = int(entry.split(' ')[1])
                    
                    if index == k+self.skip_dimensions:
                    
                        # If this key does not yet exist, create it
                        if index not in list(self.special_terms[k+self.skip_dimensions].keys()):
                            self.special_terms[k+self.skip_dimensions][index]    = {
                                'counter'   : 0,
                                'centers'   : np.asarray([]),
                                'scales'    : np.asarray([])}
                        
                        # Mark it in memory
                        self.special_terms[k+self.skip_dimensions][index]['counter'] += 1
                        
                    # Does this monotone term have cross-terms?
                    elif index != k+self.skip_dimensions: # The proposed monotone ST index is not the last argument
                    
                        # Does the dictionary have a monotone key yet?
                        if 'cross-terms' not in list(self.special_terms[k+self.skip_dimensions].keys()):
                            
                            # Create the key, if not
                            self.special_terms[k+self.skip_dimensions]['cross-terms']   = {}
                    
                        # Does this cross-term dependence have a key yet?
                        if index not in list(self.special_terms[k+self.skip_dimensions]['cross-terms'].keys()):
                            
                            # Create the key, if not
                            self.special_terms[k+self.skip_dimensions]['cross-terms'][index]    = {
                                'counter'   : 0,
                                'centers'   : np.asarray([]),
                                'scales'    : np.asarray([])}

                        # Mark it in memory
                        self.special_terms[k+self.skip_dimensions]['cross-terms'][index]['counter'] += 1

        
        return
    
    def determine_special_term_locations(self,k=None):
        
        """
        This function calculates the location and scale parameters for special
        terms in the transport map definition, specifically RBF (Radial Basis
        Functions), iRBF (Integrated Radial Basis Functions), and LET/RET (Edge
        Terms). 
                                                                        
        Position and scale parameters are assigned in the order they have been
        defined, so make sure to define a left edge term first if you want it
        to be on the left side.
        
        Variables:
        
            k - [default = None]
                [integer or None] : an integer specifying what dimension of the
                samples the 'term' corresponds to. Used to clarify with respect
                to what dimension we build this basis function
        """
        
        import copy
        
        def place_special_terms(self,dictionary):
            
            """
            A supporting function, which actually determines where the special
            terms are being placed.
            """
        
            # Find the key list
            keylist     = list(dictionary.keys())
            
            # If there is a cross-term key, ignore it.
            if 'cross-terms' in keylist:
                keylist.remove('cross-terms')
                
            # Go through all arguments with special terms
            for d in keylist:
                
                # -------------------------------------------------------------
                # One special term
                # -------------------------------------------------------------
                
                # We have only one ST
                if dictionary[d]['counter'] == 1:
            
                    # Determine the center
                    dictionary[d]['centers'] = np.asarray([np.quantile(
                        self.X[:,d],
                        q   = 0.5)])
                    
                    # Determine the scales
                    if self.ST_scale_mode == 'dynamic': # Dynamic scales
                        
                        dictionary[d]['scales'] = np.asarray([self.ST_scale_factor/2])
                        
                    elif self.ST_scale_mode == 'static': # Static scales
                        
                        dictionary[d]['scales'] = np.asarray([self.ST_scale_factor])
            
            
                # ---------------------------------------------------------
                # Multiple special terms
                # ---------------------------------------------------------
                    
                elif dictionary[d]['counter'] > 1:
                
                    # Decide where to place the special terms
                    quantiles   = np.arange(1,dictionary[d]['counter']+1,1)/(dictionary[d]['counter']+1)
                    
                    # Append an empty array, then fill it
                    scales      = np.zeros(dictionary[d]['counter'])
                    
                    # Determine the centers
                    dictionary[d]['centers'] = copy.copy(
                        np.quantile(
                            a   = self.X[:,d],
                            q   = quantiles) )

                    # Determine the scales
                    if self.ST_scale_mode == 'dynamic':
                        
                        # Otherwise, determine the scale based on relative differences
                        for i in range(dictionary[d]['counter']):
                            
                            # Left edge-case: base is half distance to next basis
                            if i == 0:
                                
                                scales[i] = \
                                    (dictionary[d]['centers'][1] - dictionary[d]['centers'][0])*self.ST_scale_factor
                                
                            # Right edge-case: base is half distance to previous basis
                            elif i == dictionary[d]['counter']-1:
                                
                                scales[i] = \
                                    (dictionary[d]['centers'][i] - dictionary[d]['centers'][i-1])*self.ST_scale_factor
                            
                            # Otherwise: base is average distance to neighbours
                            else:
                                
                                scales[i] = \
                                    (dictionary[d]['centers'][i+1] - dictionary[d]['centers'][i-1])/2*self.ST_scale_factor

                        # Copy the scales into the array
                        dictionary[d]['scales']  = copy.copy(scales)
                        
                    elif self.ST_scale_mode == 'static':

                        # Copy the scales into the array
                        dictionary[d]['scales']  = copy.copy(scales) + self.ST_scale_factor
                        
            return dictionary            
                        
        # ---------------------------------------------------------------------
        # Find the special term locations
        # ---------------------------------------------------------------------
        
        # For what terms shall we apply this update?
        if k is None:
            
            # No k is supplied, go through all components
            K   = np.arange(self.D)+self.skip_dimensions
            
        else:
            
            # A k is supplied, only apply the operation to this component
            K   = [k+self.skip_dimensions]
        
        # Go through all terms
        for k in K:
            
            # If there are cross-terms, do the same thing
            if 'cross-terms' in list(self.special_terms[k].keys()):
                
                # Write in the special term locations
                self.special_terms[k]['cross-terms']    = place_special_terms(
                    self,
                    dictionary  = copy.deepcopy(self.special_terms[k]['cross-terms']))
            
            # Write in the special term locations
            self.special_terms[k]   = place_special_terms(
                self,
                dictionary  = copy.deepcopy(self.special_terms[k]))

        # ---------------------------------------------------------------------
        # Prepare linearization thresholds, if linearization is used
        # ---------------------------------------------------------------------
            
        # Set the linearization bounds
        if self.linearization is not None:
            
            self.linearization_threshold = np.zeros((self.X.shape[-1],2))
            
            # If the linearization value specifies quantiles, calculate the linearization thresholds
            if self.linearization_specified_as_quantiles:
            
                for k in range(self.X.shape[-1]):
                    
                    self.linearization_threshold[k,0] = np.quantile(self.X[:,k], q = self.linearization)
                    self.linearization_threshold[k,1] = np.quantile(self.X[:,k], q = 1-self.linearization)
                 
            # Otherwise, directly prescribe them
            else:
                
                for k in range(self.X.shape[-1]):
                    
                    # Overwrite with static term -marked-
                    self.linearization_threshold[k,0] = -self.linearization
                    self.linearization_threshold[k,1] = +self.linearization
                
        return
    
            
    def map(self,X = None):
        
        """
        This function maps the samples X from the target distribution to the
        standard multivariate Gaussian reference distribution. If X has not 
        been provided, the samples in storage will be used instead
        
        Variables:
        
            X - [default = None]
                [None or array] : N-by-D array of the training samples used to 
                optimize the transport map, where N is the number of samples 
                and D is the number of dimensions.
        """
        
        import numpy as np
        import copy
        
        # If we have specified 
        if X is not None and self.standardize_samples:
            
            # Create a local copy of X
            X       = copy.copy(X)
            
            # Standardize the samples, if thhe user provided them
            X       -= self.X_mean
            X       /= self.X_std
            
        else:
            
            # Retrieve X from memory, create a local copy
            X       = copy.copy(self.X)
        
        # Initialize the output array
        Z   = np.zeros((X.shape[0],self.D))

        # Evaluate each of the map component functions
        for k in range(self.D):
            
            # Apply the forward map
            Z[:,k]  = copy.copy(self.s(
                x               = X,
                k               = k,
                coeffs_nonmon   = self.coeffs_nonmon[k],
                coeffs_mon      = self.coeffs_mon[k]))
            
        return Z
    
    def s(self,x, k, coeffs_nonmon = None, coeffs_mon = None):
        
        """
        This function evaluates the k-th map component.
        
        Variables:
        
            x
                [array] : N-by-D array of the training samples used to optimize
                the transport map, where N is the number of samples and D is 
                the number of dimensions. Can be None, at which point it is 
                replaced with X from storage.
                
            k
                [integer] : an integer variable defining what map component 
                is being evaluated. Corresponds to a dimension of sample space.
                
            coeffs_nonmon - [default = None]
                [vector] : a vector specifying the coefficients of the non-
                monotone part of the map component's terms, i.e., those entries
                which do not depend on x_k. This vector is replaced from 
                storage if it is not overwritten.
                
            coeffs_nonmon - [default = None]
                [vector] : a vector specifying the coefficients of the monotone 
                part of the map component's terms, i.e., those entries which do 
                not depend on x_k. This vector is replaced from storage if it 
                is not overwritten.
        """
        
        import numpy as np
        import copy
        
        # Load in values if required
        if x is None:
            
            # If x has not been specified, load it from memory
            x               = copy.copy(self.X)
            
            # Also load the matrix of nonmonotone basis function evaluations
            Psi_nonmon      = copy.copy(self.Psi_nonmon[k])
            
        else:
            
            # Evaluate the matrix of nonmonotone basis functions
            Psi_nonmon      = copy.copy(self.fun_nonmon[k](x,self))
            
        # If coefficients have not been specified, load them from storage
        if coeffs_mon is None:
            coeffs_mon      = self.coeffs_mon[k]
            
        # If coefficients have not been specified, load them from storage
        if coeffs_nonmon is None:
            coeffs_nonmon   = self.coeffs_nonmon[k]
        
        # ---------------------------------------------------------------------
        # Calculate the non-monotone part
        # ---------------------------------------------------------------------
        
        # If there are nonmonotone basis functions
        if Psi_nonmon is not None: 
            
            # Multiply them with their corresponding coefficients
            nonmonotone_part = np.dot(
                Psi_nonmon, 
                coeffs_nonmon[:,np.newaxis])[...,0]
            
        # Else, the nonmonotone part is zero
        else:
            
            nonmonotone_part = 0
        
        # ---------------------------------------------------------------------
        # Calculate the monotone part
        # ---------------------------------------------------------------------
        
        # If we use an 'integrated rectifier' approach to ensure monotonicity
        if self.monotonicity == "integrated rectifier":
        
            # Prepare the integration argument
            def integral_argument(x,y,coeffs_mon,k):
                
                # First reconstruct the full X matrix
                X_loc           = copy.copy(y)
                X_loc[:,self.skip_dimensions+k]      = copy.copy(x)
                
                # Then evaluate the Psi matrix
                Psi_mon_loc     = self.fun_mon[k](X_loc,self)
                
                # Determine the gradients
                rect_arg        = np.dot(
                    Psi_mon_loc,
                    coeffs_mon[:,np.newaxis])[...,0]
                
                # Send the rectifier argument through the rectifier
                arg     = self.rect.evaluate(rect_arg)
                
                # If there is any delta term to prevent underflow, add it
                arg     += self.delta
                
                return arg
            
            # Evaluate the integral
            monotone_part = self.GaussQuadrature(
                f               = integral_argument,
                a               = 0,
                b               = x[...,self.skip_dimensions+k],
                args            = (x,coeffs_mon,k),
                **self.quadrature_input)
        
        # If we use a 'separable monotonicity' approach to ensure monotonicity
        elif self.monotonicity == "separable monotonicity":
            
            # In the case that monotonicity is enforced through parameterization,
            # simply evaluate the monotone funciton
            monotone_part = np.dot(
                self.fun_mon[k](
                    x,
                    self),
                coeffs_mon[:,np.newaxis])[:,0]
        
        # ---------------------------------------------------------------------
        # Combine both terms
        # ---------------------------------------------------------------------
        
        # Combine the terms
        result  = copy.copy(nonmonotone_part + monotone_part)

        return result
    
    def optimize(self, K = None):
        
        """
        This function optimizes the map's component functions, seeking the
        coefficients which best map the samples to a standard multivariate
        Gaussian distribution.
        
        Variables:
        
            K - [default = None]
                [None or list] : a list of integers specifying which map 
                component functions we are optimizing. If None, the function
                optimizes all map component functions.
        
        """
        
        import numpy as np
        import copy
        
        # If we haven't specified which components should be optimized, then
        # we optimize all components
        if K is None:
            K       = np.arange(self.D)

        # With only one worker, don't parallelize. This is often cheaper due to
        # computational overhead.
        if self.workers == 1: 
            
            # The standard optimization pathway, most flexibility
            if self.monotonicity == "integrated rectifier":
                
                # Go through all map components
                for k in K:
                    
                    # Optimize this map component
                    results = self.worker_task(
                        k               = k,
                        task_supervisor = None)
                    
                    # Print optimziation progress
                    if self.verbose:
                        string = '\r'+'Progress: |'
                        string += (k+1)*'█'
                        string += (len(K)-k-1)*' '
                        string += '|'
                        print(string,end='\r')
                    
                    # Extract and store the optimized coefficients
                    self.coeffs_nonmon[k]   = copy.deepcopy(results[0])
                    self.coeffs_mon[k]      = copy.deepcopy(results[1])
                    
            # A faster, albeit less flexible optimization pathway
            elif self.monotonicity == "separable monotonicity":
                
                # Go through all map components
                for k in K:
                    
                    # Optimize this map component
                    results = self.worker_task_monotone(
                        k               = k,
                        task_supervisor = None)
                    
                    # Print optimziation progress
                    if self.verbose:
                        string = '\r'+'Progress: |'
                        string += (k+1)*'█'
                        string += (len(K)-k-1)*' '
                        string += '|'
                        print(string,end='\r')
                    
                    # Extract and store the optimized coefficients
                    self.coeffs_nonmon[k]   = copy.deepcopy(results[0])
                    self.coeffs_mon[k]      = copy.deepcopy(results[1])
        
        # If we have more than one worker, parallelize. Use with caution.
        elif self.workers > 1:  
            
            from multiprocessing import Pool, Manager
            from itertools import repeat

            # ---------------------------------------------------------------------
            # Prepare parallelization
            # ---------------------------------------------------------------------
    
            # Create the task supervisor
            manager = Manager()
            task_supervisor = manager.list([0]*len(K))
            
            # ---------------------------------------------------------------------
            # Start parallel tasks
            # ---------------------------------------------------------------------
    
            if self.monotonicity == "integrated rectifier":
                
                # For parallelization, Python seemingly cannot share functions we have
                # dynamically assembled between processes. As a consequence, we must
                # delete them, then re-create them inside the processes
                del self.fun_mon, self.fun_nonmon
                
                # Start the worker
                # We flip the order of the tasks because components farther down in the
                # transport map take longer to computer; it is computationally useful
                # to tackle these tasks first, so we don't leave the longest task last
                p = Pool(processes = self.workers)
                results = p.starmap(
                    func        = self.worker_task, 
                    iterable    = zip(
                        np.flip(K),
                        repeat(task_supervisor)) ) 
                p.close()
                p.join()
            
            elif self.monotonicity == "separable monotonicity":
                
                # For parallelization, Python seemingly cannot share functions we have
                # dynamically assembled between processes. As a consequence, we must
                # delete them, then re-create them inside the processes
                del self.fun_mon, self.fun_nonmon, self.der_fun_mon
                
                
                # Start the worker
                # We flip the order of the tasks because components farther down in the
                # transport map take longer to computer; it is computationally useful
                # to tackle these tasks first, so we don't leave the longest task last
                p = Pool(processes = self.workers)
                results = p.starmap(
                    func        = self.worker_task_monotone, 
                    iterable    = zip(
                        np.flip(K),
                        repeat(task_supervisor)) ) 
                p.close()
                p.join()
            
            # ---------------------------------------------------------------------
            # Post-process parallel task
            # ---------------------------------------------------------------------
            
            if self.verbose:
                # Make final update to the task supervisor
                string = '\r'+'Progress: |'
                for i in range(len(task_supervisor)):
                    if task_supervisor[i] == 1:     # Successful task
                        string += '█'
                    elif task_supervisor[i] == -1:  # (Partially) failed task
                        string += 'X'
                    elif task_supervisor[i] == 2:   # Successful task upon restart
                        string += 'R'
                    else:
                        string += ' '               # Unfinished task (should not occur)
                string += '|'
                print(string)
            
            # Reverse the results back into proper order
            results.reverse()
            
            # Go through all results and save the coefficients
            for k in K:
                
                # Save the coefficients
                self.coeffs_nonmon[k]   = copy.deepcopy(results[k][0])
                self.coeffs_mon[k]      = copy.deepcopy(results[k][1])
            
        # Restore the functions we previously deleted
        self.fun_mon    = []
        self.fun_nonmon = []
        if self.monotonicity == "separable monotonicity":
            self.der_fun_mon = []
            
        for k in range(self.D):
            
            # Create the function
            funstring   = "fun_mon_"+str(k)
            exec(self.fun_mon_strings[k].replace("fun",funstring), globals())
            exec("self.fun_mon.append(copy.deepcopy("+funstring+"))")
    
            # Create the function
            funstring   = "fun_nonmon_"+str(k)
            exec(self.fun_nonmon_strings[k].replace("fun",funstring), globals())
            exec("self.fun_nonmon.append(copy.deepcopy("+funstring+"))")
            
            if self.monotonicity == "separable monotonicity":
            
                # Create the function
                funstring   = "der_fun_mon_"+str(k)
                exec(self.der_fun_mon_strings[k].replace("fun",funstring), globals())
                exec("self.der_fun_mon.append(copy.deepcopy("+funstring+"))")
    
        return
    
    
    def worker_task_monotone(self,k,task_supervisor):
        
        """
        This function provides the optimization task for the k-th map component
        function to a worker (if parallelization is used), or applies it in 
        sequence (if no parallelization is used). This specific function only
        becomes active if monotonicity = 'separable monotonicity'.
        
        Variables:
        
            k
                [integer] : an integer variable defining what map component 
                is being evaluated. Corresponds to a dimension of sample space.
                
            task supervisor
                [list] : a shared list which informs the main process how many
                optimization tasks have already been computed. This list should
                not be specified by the user, it only serves to provide 
                information about the optimization progress.
        """
        
        import numpy as np
        from scipy.optimize import minimize
        import copy
        
        # -----------------------------------------------------------------
        # Prepare task
        # -----------------------------------------------------------------
        
        if task_supervisor is not None and self.verbose:
        
            # Print multiprocessing progress
            string = '\r'+'Progress: |'
            for i in range(len(task_supervisor)):
                if task_supervisor[i] == 1:
                    string += '█'
                elif task_supervisor[i] == -1:
                    string += 'X'
                elif task_supervisor[i] == 2:
                    string += 'R'
                else:
                    string += ' '
            string += '|'
            if self.workers == 1:
                print(string,end='\r')
        
        # Create local copies of the nonmonotone and monotone basis function's
        # coefficients.
        coeffs_nonmon   = copy.copy(self.coeffs_nonmon[k])
        coeffs_mon      = copy.copy(self.coeffs_mon[k])
            
        # ---------------------------------------------------------------------
        # Define special objective for the monotone function
        # ---------------------------------------------------------------------
        
        # If we do not regularize
        if self.regularization is None:
            
            # -----------------------------------------------------------------
            # No regularization, use standard objective
            # -----------------------------------------------------------------
        
            # Make a QR decomposition of the nonmonotone basis function matrix
            Q,R         = np.linalg.qr(self.Psi_nonmon[k], mode='reduced')
            
            # Calculate the A_sqrt term
            A_sqrt  = self.Psi_mon[k] - np.linalg.multi_dot((Q,Q.T,self.Psi_mon[k]))
            
            # Get the ensemble size
            N           = self.X.shape[0]
            
            # Calculate the A term
            A           = np.dot(A_sqrt.T,A_sqrt)/N
                
            # The optimization objective for the monotone coefficients
            def fun_mon_objective(coeffs_mon,A,k,all_outputs=True):
                
                # -------------------------------------------------------------
                # Determine objective
                # -------------------------------------------------------------
                
                b      = self.delta*np.sum(A,axis=-1)
        
                Ax     = np.dot(
                    A,
                    coeffs_mon[:,np.newaxis])
        
                dS     = np.dot(
                    self.der_Psi_mon[k],
                    coeffs_mon[:,np.newaxis]) + np.sum(self.der_Psi_mon[k],axis=-1)[:,np.newaxis]*self.delta
                
                logdS   = np.log(dS)

                objective   = np.dot(coeffs_mon[np.newaxis,:],Ax)[0,0]/2 - np.sum(logdS)/N + np.inner(coeffs_mon,b)
    
                if all_outputs:
    
                    # -------------------------------------------------------------
                    # Determine Jacobian
                    # -------------------------------------------------------------
            
                    dPsi_dS = self.der_Psi_mon[k]/dS
                    
                    grad    = Ax[:,0] - np.sum(dPsi_dS,axis=0)/N + b
                    
                    # -------------------------------------------------------------
                    # Determine Hessian
                    # -------------------------------------------------------------
            
                    hess    = A + np.dot(dPsi_dS.T,dPsi_dS)/N
                
                    return objective, grad, hess
                
                else:
                    
                    return objective

        # If we regularize
        elif self.regularization.lower() == 'l2':
            
            # -----------------------------------------------------------------
            # L2 regularization, use alternative objective
            # -----------------------------------------------------------------
            
            # Get the ensemble size
            N           = self.X.shape[0]
            
            # Step 1: Calculate basis for the supporting variable A
            A   = np.linalg.multi_dot((
                np.linalg.inv(
                    np.dot(
                        self.Psi_nonmon[k].T,
                        self.Psi_nonmon[k]) + self.regularization_lambda*np.identity(self.Psi_nonmon[k].shape[-1])),
                self.Psi_nonmon[k].T,
                self.Psi_mon[k]))
            
            # Step 2: Aggregate
            A   = np.dot((
                self.Psi_mon[k] - np.dot(
                    self.Psi_nonmon[k],
                    A)).T,
                self.Psi_mon[k] - np.dot(
                    self.Psi_nonmon[k],
                    A) ) / 2 + \
                self.regularization_lambda * (
                np.dot(
                    A.T, 
                    A) + np.identity(A.shape[-1]))
                
            # Create the objective function
            def fun_mon_objective(coeffs_mon,A,k,all_outputs=True):
                
                # -------------------------------------------------------------
                # Determine objective
                # -------------------------------------------------------------
                
                b      = self.delta*np.sum(A,axis=-1)
        
                Ax     = np.dot(
                    A,
                    coeffs_mon[:,np.newaxis])
        
                dS     = np.dot(
                    self.der_Psi_mon[k],
                    coeffs_mon[:,np.newaxis]) + np.sum(self.der_Psi_mon[k],axis=-1)[:,np.newaxis]*self.delta
                
                logdS   = np.log(dS)


                objective   = np.dot(coeffs_mon[np.newaxis,:],Ax)[0,0]/2 - np.sum(logdS)/N + np.inner(coeffs_mon,b)
    
                if all_outputs:
    
                    # -------------------------------------------------------------
                    # Determine Jacobian
                    # -------------------------------------------------------------
            
                    dPsi_dS = self.der_Psi_mon[k]/dS
                    
                    grad    = Ax[:,0] - np.sum(dPsi_dS,axis=0)/N + b
                    
                    # -------------------------------------------------------------
                    # Determine Hessian
                    # -------------------------------------------------------------
            
                    hess    = A + np.dot(dPsi_dS.T,dPsi_dS)/N
                
                    return objective, grad, hess
                
                else:
                    
                    return objective
    
        # ---------------------------------------------------------------------
        # Call the optimization routine
        # ---------------------------------------------------------------------

        # Specify the optimization bounds
        bounds      = []
        for idx in range(len(self.optimization_constraints_lb[k])):
            bounds.append(
                [self.optimization_constraints_lb[k][idx], #-marked- used to have +1E-8
                 self.optimization_constraints_ub[k][idx]])
        
        # Solve the optimization problem
        opt     = minimize(
            fun     = fun_mon_objective,
            method  = 'L-BFGS-B',
            x0      = coeffs_mon,
            jac     = True,
            bounds  = bounds,
            args    = (A,k))
        
        # Extract the optimal coefficients
        coeffs_mon  = opt.x
        
        # ---------------------------------------------------------------------
        # Post-process the optimization results
        # ---------------------------------------------------------------------

        # Update the task_supervisor and print the update
        if task_supervisor is not None and self.verbose:
            
            # If optimization was a success, mark it as such
            task_supervisor[k] = 1
        
            # Print multiprocessing progress
            string = '\r'+'Progress: |'
            for i in range(len(task_supervisor)):
                if task_supervisor[i] == 1:
                    string += '█'
                elif task_supervisor[i] == -1:
                    string += 'X'
                elif task_supervisor[i] == 2:
                    string += 'R'
                else:
                    string += ' '
            string += '|'
            if self.workers == 1:
                print(string,end='\r')
            
        # ---------------------------------------------------------------------
        # With the monotone coefficients found, calculate the nonmonotone coeffs
        # ---------------------------------------------------------------------
        
        if self.regularization is None:
        
            # In the standard formulation, use the QR decomposition to calculate
            # the nonmonotone coefficients
            coeffs_nonmon   = \
                -np.linalg.multi_dot((
                    np.linalg.inv(R),
                    Q.T,
                    self.Psi_mon[k],
                    coeffs_mon[:,np.newaxis]))[:,0]
                
        elif self.regularization.lower() == 'l2':
            
            coeffs_nonmon   = \
                - np.linalg.multi_dot((
                    np.linalg.inv(
                        np.dot(self.Psi_nonmon[k].T,
                                self.Psi_nonmon[k]) + \
                    2*self.regularization_lambda*np.identity(self.Psi_nonmon[k].shape[-1])),
                    np.dot(self.Psi_nonmon[k].T,
                            self.Psi_mon[k]),
                    coeffs_mon[:,np.newaxis]))[:,0]
        
        # Return both optimized coefficients
        return (coeffs_nonmon,coeffs_mon)
    
    def worker_task(self,k,task_supervisor):
        
        """
        This function provides the optimization task for the k-th map component
        function to a worker (if parallelization is used), or applies it in 
        sequence (if no parallelization is used). This specific function only
        becomes active if monotonicity = 'integrated rectifier'.
        
        Variables:
        
            k
                [integer] : an integer variable defining what map component 
                is being evaluated. Corresponds to a dimension of sample space.
                
            task supervisor
                [list] : a shared list which informs the main process how many
                optimization tasks have already been computed. This list should
                not be specified by the user, it only serves to provide 
                information about the optimization progress.
        """
        
        from scipy.optimize import minimize
        import copy
        
        # -----------------------------------------------------------------
        # Prepare task
        # -----------------------------------------------------------------
        
        if task_supervisor is not None and self.verbose:
        
            # Print multiprocessing progress
            string = '\r'+'Progress: |'
            for i in range(len(task_supervisor)):
                if task_supervisor[i] == 1:
                    string += '█'
                elif task_supervisor[i] == -1:
                    string += 'X'
                elif task_supervisor[i] == 2:
                    string += 'R'
                else:
                    string += ' '
            string += '|'
            print(string,end='\r')
        
        # Assemble the theta vector we are optimizing
        coeffs          = np.zeros(len(self.coeffs_nonmon[k]) + len(self.coeffs_mon[k]))
        div             = len(self.coeffs_nonmon[k]) # Divisor for the vector
        
        # Write in the coefficients
        coeffs[:div]    = copy.copy(self.coeffs_nonmon[k])
        coeffs[div:]    = copy.copy(self.coeffs_mon[k])
        
        # ---------------------------------------------------------------------
        # Re-create the functions
        # ---------------------------------------------------------------------
        
        if self.workers > 1:
        
            # Restore the functions we previously deleted
            self.fun_mon    = []
            self.fun_nonmon = []
            for i in range(self.D):
                
                # Create the function
                funstring   = "fun_mon_"+str(i)
                exec(self.fun_mon_strings[i].replace("fun",funstring), globals())
                exec("self.fun_mon.append(copy.deepcopy("+funstring+"))")
        
                # Create the function
                funstring   = "fun_nonmon_"+str(i)
                exec(self.fun_nonmon_strings[i].replace("fun",funstring), globals())
                exec("self.fun_nonmon.append(copy.deepcopy("+funstring+"))")

        # ---------------------------------------------------------------------
        # Call the optimization routine
        # ---------------------------------------------------------------------
        
        # Minimize the objective function
        opt     = minimize(
            method  = 'BFGS',#'L-BFGS-B',
            fun     = self.objective_function,
            jac     = self.objective_function_jacobian,
            x0      = coeffs,
            args    = (k,div))
        
        # ---------------------------------------------------------------------
        # Post-process the optimization results
        # ---------------------------------------------------------------------

        # Retrieve the optimized coefficients
        coeffs_opt      = copy.copy(opt.x)
        
        # Separate them into coefficients for monotone and nonmonotone parts
        coeffs_nonmon   = coeffs_opt[:div]
        coeffs_mon      = coeffs_opt[div:]
        
        if task_supervisor is not None and self.verbose:
        
            # If optimization was a success, mark it as such
            if opt.success:
                
                # '1' represents initial sucess ('█')
                task_supervisor[k] = 1
                
            else:
                
                # '-1' represents failure ('X')
                task_supervisor[k] = -1
                
            # Print multiprocessing progress
            string = '\r'+'Progress: |'
            for i in range(len(task_supervisor)):
                if task_supervisor[i] == 1:
                    string += '█'
                elif task_supervisor[i] == -1:
                    string += 'X'
                elif task_supervisor[i] == 2:
                    string += 'R'
                else:
                    string += ' '
            string += '|'
            print(string,end='\r')
            
        # Return both optimized coefficients
        return (coeffs_nonmon,coeffs_mon)
    
    def objective_function(self, coeffs, k, div = 0):
        
        """
        This function evaluates the objective function used in the optimization
        of the map's component functions.
        
        Variables:
        
            coeffs
                [vector] : a vector containing the coefficients for both the
                nonmonotone and monotone terms of the k-th map component 
                function. Is replaced for storage is specified as None.
                
            k
                [integer] : an integer variable defining what map component 
                is being evaluated. Corresponds to a dimension of sample space.
                
            div - [default = 0]
                [integer] : an integer specifying where the cutoff between the
                nonmonotone and monotone coefficients in 'coeffs' is.
        """
        
        import numpy as np
        import copy
        import scipy.special
        
        # Partition the coefficient vector, if necessary
        if coeffs is not None:
            # Separate the vector into nonmonotone and monotone coefficients
            coeffs_nonmon   = copy.copy(coeffs[:div])
            coeffs_mon      = copy.copy(coeffs[div:])
        else:
            if self.verbose:
                print('loading')
            # Otherwise, load them from object
            coeffs_nonmon   = copy.copy(self.coeffs_nonmon[k])
            coeffs_mon      = copy.copy(self.coeffs_mon[k])
        
        # ---------------------------------------------------------------------
        # First part: How close is the ensemble mapped to zero?
        # ---------------------------------------------------------------------
        
        # Map the samples to the reference marginal
        map_result  = self.s(
            x               = None, 
            k               = k, 
            coeffs_nonmon   = coeffs_nonmon, 
            coeffs_mon      = coeffs_mon)
        
        # Check how close these samples are to the origin
        objective   = 1/2*map_result**2
        
        # print(objective)
        
        # ---------------------------------------------------------------------
        # Second part: How much is the ensemble inflated?
        # ---------------------------------------------------------------------
        
        Psi_mon     = self.fun_mon[k](self.X,self)
        
        # Determine the gradients of the polynomial functions
        monotone_part_der = np.dot(
            Psi_mon,
            coeffs_mon[:,np.newaxis])[...,0]
        
        # Evaluate the logarithm of the recetified monotone part        
        obj         = self.rect.logevaluate(monotone_part_der)
        
        # Subtract this from the objective
        objective   -= obj
        
        # ---------------------------------------------------------------------
        # Average the objective function
        # ---------------------------------------------------------------------
        
        # Now summarize the contributions and take their average
        objective   = np.mean(objective) 
        
        # ---------------------------------------------------------------------
        # Add regularization, if desired
        # ---------------------------------------------------------------------
        
        if self.regularization is not None:
            
            # A scalar regularization was specified
            if type(self.regularization) == str:
                
                if self.regularization.lower() == 'l1':
                    
                    # Regularization_lambda is identical for all parameters
                    if np.isscalar(self.regularization_lambda):
                        
                        # Add l1 regularization for all coefficients
                        objective   += self.regularization_lambda*np.sum(np.abs(coeffs_mon))
                        objective   += self.regularization_lambda*np.sum(np.abs(coeffs_nonmon))
                        
                    elif type(self.regularization_lambda) == list:
                        
                        # Add l1 regularization for all coefficients
                        objective   += np.sum(self.regularization_lambda[k][div:]*np.abs(coeffs_mon))
                        objective   += np.sum(self.regularization_lambda[k][:div]*np.abs(coeffs_nonmon))
                        
                    else:
                        
                        raise ValueError("Data type of regularization_lambda not understood. Must be either scalar or list.")
                    
                elif self.regularization.lower() == 'l2':
                    
                    # Regularization_lambda is identical for all parameters
                    if np.isscalar(self.regularization_lambda):
                    
                        # Add l2 regularization for all coefficients
                        objective   += self.regularization_lambda*np.sum(coeffs_mon**2)
                        objective   += self.regularization_lambda*np.sum(coeffs_nonmon**2)
                        
                    elif type(self.regularization_lambda) == list:
                        
                        # Add l1 regularization for all coefficients
                        objective   += np.sum(self.regularization_lambda[k][div:]*coeffs_mon**2)
                        objective   += np.sum(self.regularization_lambda[k][:div]*coeffs_nonmon**2)
                        
                    else:
                        
                        raise ValueError("Data type of regularization_lambda not understood. Must be either scalar or list.")
                    
                else:
                    
                    raise ValueError("regularization_type must be either 'l1' or 'l2'.")

            else:
                
                raise ValueError("The variable 'regularization' must be either None, 'l1', or 'l2'.")
        
        return objective
    
    def objective_function_jacobian(self, coeffs, k, div = 0):
        
        """
        This function evaluates the derivative of the objective function used 
        in the optimization of the map's component functions.
        
        Variables:
        
            coeffs
                [vector] : a vector containing the coefficients for both the
                nonmonotone and monotone terms of the k-th map component 
                function. Is replaced for storage is specified as None.
                
            k
                [integer] : an integer variable defining what map component 
                is being evaluated. Corresponds to a dimension of sample space.
                
            div - [default = 0]
                [integer] : an integer specifying where the cutoff between the
                nonmonotone and monotone coefficients in 'coeffs' is.
        """
        
        import numpy as np
        import copy
        
        # Partition the coefficient vector, if necessary
        if coeffs is not None:
            # Separate the vector into nonmonotone and monotone coefficients
            coeffs_nonmon   = copy.copy(coeffs[:div])
            coeffs_mon      = copy.copy(coeffs[div:])
        else:
            # Otherwise, load them from object
            coeffs_nonmon   = copy.copy(self.coeffs_nonmon[k])
            coeffs_mon      = copy.copy(self.coeffs_mon[k])
        
        # =====================================================================
        # Prepare term 1
        # =====================================================================
        
        # First, handle the scalar
        term_1_scalar   = self.s(
            x               = None, 
            k               = k, 
            coeffs_nonmon   = coeffs_nonmon, 
            coeffs_mon      = coeffs_mon)
        
        
        # Define the integration argument
        def integral_argument_term1_jac(x,coeffs_mon,k): 
            
            # First reconstruct the full X matrix
            X_loc           = copy.copy(self.X)
            X_loc[:,self.skip_dimensions+k]      = copy.copy(x)
            
            # Calculate the local basis function matrix
            Psi_mon_loc     = self.fun_mon[k](X_loc,self)
                
            # Determine the gradients
            rec_arg         = np.dot(
                Psi_mon_loc,
                coeffs_mon[:,np.newaxis])[...,0]
            
            objective = self.rect.evaluate_dfdc(
                f           = rec_arg, 
                dfdc        = Psi_mon_loc)
         
            return objective
                      
        # Add the integration
        term_1_vector_monotone  = self.GaussQuadrature(
            f       = integral_argument_term1_jac,
            a       = 0,
            b       = self.X[:,self.skip_dimensions+k],
            args    = (coeffs_mon,k),
            **self.quadrature_input)
        
        # If we have non-monotone terms, consider them
        if self.Psi_nonmon[k] is not None:
            
            # Evaluate the non-monotone vector term
            term_1_vector_nonmonotone   = copy.copy(self.Psi_nonmon[k])
            
            # Stack the results together
            term_1_vector   = np.column_stack((
                term_1_vector_nonmonotone,
                term_1_vector_monotone))
            
        else:
            
            # If we have no non-monotone terms, the vector is only composed of
            # monotone coefficients
            term_1_vector   = term_1_vector_monotone
        
        # Combine to obtain the full term 1
        term_1 = np.einsum(
            'i,ij->ij',
            term_1_scalar,
            term_1_vector)
        
        # =====================================================================
        # Prepare term 2
        # =====================================================================

        # Create term_2
        # https://www.wolframalpha.com/input/?i=derivative+of+log%28f%28c%29%29+wrt+c

        rec_arg     = np.dot(
            self.Psi_mon[k],
            coeffs_mon[:,np.newaxis])[...,0] # This is dfdk
        
        numer       = self.rect.evaluate_dfdc(
            f       = rec_arg,
            dfdc    = self.Psi_mon[k])
        
        denom       = 1/(self.rect.evaluate(rec_arg) + self.delta)
        
        term_2 = np.einsum(
            'ij,i->ij',
            numer,
            denom)
        
        if div > 0:
            
            # If we have non-monotone terms, expand the term accordingly
            term_2  = np.column_stack((
                np.zeros((term_2.shape[0],div)),
                term_2))
            
        # =====================================================================
        # Combine both terms
        # =====================================================================

        objective   = np.mean(
            term_1 - \
            term_2 ,axis=0)
            
        # ---------------------------------------------------------------------
        # Add regularization, if desired
        # ---------------------------------------------------------------------
        
        if self.regularization is not None:
            
            # A scalar regularization was specified
            if type(self.regularization) == str:
                
                if self.regularization.lower() == 'l1':
                    

                    # Regularization_lambda is identical for all parameters
                    if np.isscalar(self.regularization_lambda):
                        
                        # Add l1 regularization for all coefficients
                        term        = np.asarray(
                            list(self.regularization_lambda*np.sign(coeffs_nonmon)) + \
                            list(self.regularization_lambda*np.sign(coeffs_mon)))
                        
                    elif type(self.regularization_lambda) == list:
                        
                        # Add l1 regularization for all coefficients
                        term        = np.asarray(
                            list(np.sign(coeffs_nonmon)) + \
                            list(np.sign(coeffs_mon))) * self.regularization_lambda[k]
                        
                    else:
                        
                        raise ValueError("Data type of regularization_lambda not understood. Must be either scalar or list.")
                    
                    objective   += term
                    
                elif self.regularization.lower() == 'l2':
                    
                    # Regularization_lambda is identical for all parameters
                    if np.isscalar(self.regularization_lambda):
                        
                        # Add l2 regularization for all coefficients
                        term        = np.asarray(
                            list(self.regularization_lambda*2*coeffs_nonmon) + \
                            list(self.regularization_lambda*2*coeffs_mon))
                        
                    elif type(self.regularization_lambda) == list:
                        
                        # Add l2 regularization for all coefficients
                        term        = np.asarray(
                            list(2*coeffs_nonmon) + \
                            list(2*coeffs_mon)) * self.regularization_lambda[k]
                        
                    else:
                        
                        raise ValueError("Data type of regularization_lambda not understood. Must be either scalar or list.")
                    
                    objective   += term
                    
                else:
                    
                    raise ValueError("regularization_type must be either 'l1' or 'l2'.")
        
            else:
                
                raise ValueError("The variable 'regularization' must be either None, 'l1', or 'l2'.")
        
        return objective
    
    

    def inverse_map(self, Z, X_star = None):
        
        """
        This function evaluates the inverse transport map, mapping samples from
        a multivariate standard Gaussian back to the target distribution. If
        X_precalc is specified, the map instead evaluates a conditional of the 
        target distribution given X_precalc. The function assumes any 
        precalculated output are the FIRST dimensions of the total output. If 
        X_precalc is specified, its dimensions and the input dimensions must 
        sum to the full dimensionality of sample space.
        
        Variables:
        
            Z
                [array] : N-by-D or N-by-(D-E) array of reference distribution
                samples to be mapped to the target distribution, where N is the 
                number of samples, D is the number of target distribution 
                dimensions, and E the number of pre-specified dimenions (if 
                X_precalc is specified).
                
            X_star - [default = None]
                [None or array] : N-by-E array of samples in the space of the
                target distribution, used to condition the lower D-E dimensions
                during the inversion process.
        """
        
        import numpy as np
        import copy
        
        # Create a local copy of Z to prevent overwriting the input
        Z   = copy.copy(Z)
        
        # Extract number of samples
        N   = Z.shape[0]
        
        # =====================================================================
        # No X_star was provided
        # =====================================================================
        
        if X_star is None: # Yes
        
            # Initialize the output ensemble
            X   = np.zeros((N,self.skip_dimensions + self.D))
            
            # Go through all dimensions
            for k in np.arange(0,self.D,1):
                
                if self.alternate_root_finding and self.monotonicity.lower() == 'separable monotonicity':
                
                    X      = self.vectorized_root_search_alternate(
                        Zk          = Z[:,k],
                        X           = X,
                        k           = k)
                    
                else:

                    X      = self.vectorized_root_search_bisection(
                        Zk          = Z[:,k],
                        X           = X,
                        k           = k)
                
            # If we standardized the samples, undo the standardization
            if self.standardize_samples:
                
                X   *= self.X_std
                X   += self.X_mean
                
        # =====================================================================
        # X_star was provided, and matches the reduced map definition
        # =====================================================================     
           
        if X_star is not None:
        
            if X_star.shape[-1] == self.skip_dimensions: # Yes
    
                # Initialize the output ensemble
                X   = np.zeros((N,self.skip_dimensions + self.D))
                
                # If we standardize the samples, we must also standardize the
                # precalculated values first
                X[:,:self.skip_dimensions]  = copy.copy(X_star)
                
                if self.standardize_samples:
                    
                    X[:,:self.skip_dimensions]  -= self.X_mean[:self.skip_dimensions]
                    X[:,:self.skip_dimensions]  /= self.X_std[:self.skip_dimensions]
                        
                # Go through all dimensions
                for k in np.arange(0,self.D,1):
                    
                    if self.alternate_root_finding and self.monotonicity.lower() == 'separable monotonicity':
                    
                        X      = self.vectorized_root_search_alternate(
                            Zk          = Z[:,k],
                            X           = X,
                            k           = k)
                        
                    else:
                        
                        X      = self.vectorized_root_search_bisection(
                            Zk          = Z[:,k],
                            X           = X,
                            k           = k)
                    
                # If we standardized the samples, undo the standardization
                if self.standardize_samples:
                    
                    X   *= self.X_std
                    X   += self.X_mean
                    
                    
            # =================================================================
            # A full map was defined, but so were precalculated values
            # =================================================================
                    
            elif self.skip_dimensions == 0 and X_star is not None: 
                
                # Create a local copy of skip_dimensions
                skip_dimensions     = X_star.shape[-1]
                D                   = skip_dimensions + Z.shape[-1]
                
                # Initialize the output ensemble
                X   = np.zeros((N,D))
                
                # If we standardize the samples, we must also standardize the
                # precalculated values first
                X[:,:skip_dimensions]  = copy.copy(X_star)
                
                if self.standardize_samples:
                    
                    # Standardize the precalculated samples for the map
                    X[:,:skip_dimensions]  -= self.X_mean[:skip_dimensions]
                    X[:,:skip_dimensions]  /= self.X_std[:skip_dimensions]
                        
                # Go through all dimensions
                for i,k in enumerate(np.arange(skip_dimensions,D,1)):
                    
                    if self.alternate_root_finding and self.monotonicity.lower() == 'separable monotonicity':
                    
                        X      = self.vectorized_root_search_alternate(
                            Zk          = Z[:,i],
                            X           = X,
                            k           = k)
                        
                    else:
                        
                        X      = self.vectorized_root_search_bisection(
                            Zk          = Z[:,i],
                            X           = X,
                            k           = k)
                    
                # If we standardized the samples, undo the standardization
                if self.standardize_samples:
                    
                    X   *= self.X_std
                    X   += self.X_mean
                    
        return X[:,self.skip_dimensions:]
    
    def vectorized_root_search_bisection(self, X, Zk, k, max_iterations = 100, 
        threshold = 1E-9, start_distance = 2):
        
        
        """
        This function searches for the roots of the k-th map component through
        bisection. It is called in the inverse_map function.
        
        Variables:
        
            X
                [array] : N-by-k array of samples inverted so far, where the 
                k-th column still contains the reference samples used as a 
                residual in the root finding process
                
            Zk
                [vector] : a vector containing the target values in the k-th 
                dimension, for which the root finding algorithm must solve.
                
            k
                [integer] : an integer variable defining what map component 
                is being evaluated. Corresponds to a dimension of sample space.
                
            max_iterations - [default = 100]
                [integer] : number of function calls before the algorithm stops
                continuing the root search to avoid becoming stuck in an 
                endless loop.
                
            threshold - [default = 1E-9]
                [float] : threshold value below which the algorithm assumes the
                root finding problem to be solves.
                
            start_distance - [default = 2]
                [integer] : starting distance from the origin for the interval
                edges used for bisection. This window can be moved by the 
                algorithm should the root not lie within.                
                
        """
        
        import numpy as np
        import copy
        import pickle
        
        # Extract number of particles
        N               = X.shape[0]
        
        # Check whether samples have been marked for removal
        indices         = np.arange(N)      # Indices of all particles
        failure         = np.isnan(X[:,self.skip_dimensions + k])  # Particles marked for removal
        indices         = indices[~failure] # Kill the associated indices
        
        # Initialize the start bisection points
        bsct_pts        = np.zeros((N,2))
        bsct_pts[:,0]   = -np.ones(N)*start_distance
        bsct_pts[:,1]   = +np.ones(N)*start_distance
        
        bsct_out        = np.zeros((N,2))
        
        # Calculate the initial bracket
        X[indices,self.skip_dimensions + k]    = bsct_pts[indices,0]    
        bsct_out[indices,0] = self.s(
            x           = X[indices,:],
            k           = k) - Zk[indices]
        X[indices,self.skip_dimensions + k]    = bsct_pts[indices,1]
        bsct_out[indices,1] = self.s(
            x           = X[indices,:],
            k           = k) - Zk[indices]
        
        # Sort the bsct_pts so that bsct_out is increasing
        for n in indices:
            if bsct_out[n,0] > bsct_out[n,1]:
                
                dummy           = bsct_out[n,0]
                bsct_out[n,0]   = bsct_out[n,1]
                bsct_out[n,1]   = dummy
                
                dummy           = bsct_pts[n,0]
                bsct_pts[n,0]   = bsct_pts[n,1]
                bsct_pts[n,1]   = dummy
                
        
        # =====================================================================
        # Shift windows
        # =====================================================================  
        
        # An initial proposal for the windows has been made. If zero is not
        # between the two bsct_pts, we must shift the window
        
        # Create a copy of indices
        shiftindices    = copy.copy(indices)
        
        # Where the product has different signs, zero is in-between
        failure         = np.where(np.prod(bsct_out[shiftindices,:],axis=1) > 0 )[0]
        shiftindices    = shiftindices[failure]
        
        # While
        while len(shiftindices) > 0:
            
            # Re-sort the windows if necessary
            for n in shiftindices:
                if bsct_out[n,0] > bsct_out[n,1]:
                    
                    dummy           = bsct_out[n,0]
                    bsct_out[n,0]   = bsct_out[n,1]
                    bsct_out[n,1]   = dummy
                    
                    dummy           = bsct_pts[n,0]
                    bsct_pts[n,0]   = bsct_pts[n,1]
                    bsct_pts[n,1]   = dummy
            
            # Find out the sign of the points which were NOT successful
            # sign_failure    = np.sign(np.sum(bsct_out[shiftindices,:],axis=1))
            sign_failure    = np.sign(bsct_out[shiftindices,0])
            
            # This difference tells us how much we must shift X to move RIGHT
            difference  = np.diff(bsct_pts[shiftindices,:], axis = 1)[:,0]
            
            # For positive signs, shift the window to the LEFT bound
            failure_pos = np.where(sign_failure > 0)[0]
            bsct_pts[shiftindices[failure_pos],1]   = copy.copy(bsct_pts[shiftindices[failure_pos],0])
            bsct_pts[shiftindices[failure_pos],0]   -= difference[failure_pos]*2
            
            # Re-simulate that
            bsct_out[shiftindices[failure_pos],1]   = copy.copy(bsct_out[shiftindices[failure_pos],0])
            X[shiftindices[failure_pos],self.skip_dimensions + k]          = copy.copy(bsct_pts[shiftindices[failure_pos],0])
            bsct_out[shiftindices[failure_pos],0]   = copy.copy(self.s(
                x       = X[shiftindices[failure_pos],:],
                k       = k) - Zk[shiftindices[failure_pos]])
            
            # For negative signs, shift the window to the RIGHT bound
            failure_neg = np.where(sign_failure < 0)[0]
            bsct_pts[shiftindices[failure_neg],0]   = copy.copy(bsct_pts[shiftindices[failure_neg],1])
            bsct_pts[shiftindices[failure_neg],1]   += difference[failure_neg]*2
            
            # Re-simulate that
            bsct_out[shiftindices[failure_neg],0]   = copy.copy(bsct_out[shiftindices[failure_neg],1])
            X[shiftindices[failure_neg],self.skip_dimensions + k]          = copy.copy(bsct_pts[shiftindices[failure_neg],1])
            bsct_out[shiftindices[failure_neg],1]   = copy.copy(self.s(
                x       = X[shiftindices[failure_neg],:],
                k       = k) - Zk[shiftindices[failure_neg]])
            
            # Where the product has different signs, zero is in-between
            failure         = np.where(np.prod(bsct_out[shiftindices,:],axis=1) > 0 )[0]
            shiftindices    = shiftindices[failure]
            

        # =====================================================================
        # Start the actual root search
        # =====================================================================
            
        # Prepare iteration counter
        itr_counter = 0
        
        # Start optimization loop
        while np.sum(indices) > 0 and itr_counter < max_iterations:
            
            itr_counter     += 1
            
            # Propose bisection
            mid_pt      = np.mean(bsct_pts[indices,:],axis=1)
            
            # Calculate the biscetion point output
            X[indices,self.skip_dimensions + k]      = mid_pt
            mid_out     = self.s(
                x       = X[indices,:],
                k       = k) - Zk[indices]
            
            # Set the Lower or upper boundary depending on the sign of mid_out
            below   = np.where(mid_out < 0)[0]
            above   = np.where(mid_out > 0)[0]
            
            # bsct_pts[indices[below],maxidx] = copy.copy(bsct_pts[indices[below],minidx])
            bsct_pts[indices[below],0] = copy.copy(mid_pt[below])
            
            # bsct_pts[indices[above],minidx] = copy.copy(bsct_pts[indices[above],maxidx])
            bsct_pts[indices[above],1] = copy.copy(mid_pt[above])
            
            not_converged   = np.where(np.abs(mid_out) > threshold)
            indices         = indices[not_converged]    
            
        if itr_counter == max_iterations and self.verbose:
            
            
            print('WARNING: root search for particles '+str(indices)+' stopped'+\
                  ' at maximum iterations.')
            

        return X
    
    def vectorized_root_search_alternate(self, X, Zk, k, start_distance = 10,
        resolution = 1001):
        
        
        """
        This function is an alternative root search routine, not based on
        bisection but interpolation. 
        
        Only used for "separable monotonicity"
        
        Variables:
        
            X
                [array] : N-by-k array of samples inverted so far, where the 
                k-th column still contains the reference samples used as a 
                residual in the root finding process
                
            Zk
                [vector] : a vector containing the target values in the k-th 
                dimension, for which the root finding algorithm must solve.
                
            k
                [integer] : an integer variable defining what map component 
                is being evaluated. Corresponds to a dimension of sample space.
                
            max_iterations - [default = 100]
                [integer] : number of function calls before the algorithm stops
                continuing the root search to avoid becoming stuck in an 
                endless loop.
                
            threshold - [default = 1E-9]
                [float] : threshold value below which the algorithm assumes the
                root finding problem to be solves.
                
            start_distance - [default = 2]
                [integer] : starting distance from the origin for the interval
                edges used for bisection. This window can be moved by the 
                algorithm should the root not lie within.                
                
        """
        
        import numpy as np
        import copy
        from scipy.interpolate import interp1d
        
        # Create a local copy
        X       = copy.copy(X)
        
        # ---------------------------------------------------------------------
        # Step 1: For separable monotonicity, all non-monotone terms are just
        # constant offsets. So let's just calculate that once
        
        offset = np.dot(
            self.fun_nonmon[k](
                copy.copy(X),
                self),
            self.coeffs_nonmon[k][:,np.newaxis])[:,0]

        # ---------------------------------------------------------------------
        # Step 2: Evaluate the forward map
        pts             = np.linspace(-start_distance,start_distance,resolution)

        # Create a fake X vector for the evaluation points
        fakeX           = np.zeros((resolution,X.shape[-1]))
        fakeX[:,self.skip_dimensions + k] = copy.copy(pts)
        
        # Evaluate the monotone map part
        out = np.dot(
            self.fun_mon[k](
                fakeX,
                self),
            self.coeffs_mon[k][:,np.newaxis])[:,0]
            
        # ---------------------------------------------------------------------
        # Step 3: Create a 1D interpolator
        itp     = interp1d(
            x           = out,
            y           = pts,
            fill_value  = "extrapolate")
        
        # ---------------------------------------------------------------------
        # Step 4: Evaluate the 1D interpolator
        
        # Find the target values
        target  = - offset + Zk
        
        # prevent undue extrapolation
        if self.root_search_truncation:
            target[target < np.min(out)] = np.min(out)
            target[target > np.max(out)] = np.max(out)
        
        # Find the target root
        result  = itp(target)
        
        # Save the result    
        X[:,self.skip_dimensions + k] = copy.copy(result)

        return X
    

    def GaussQuadrature(self, f, a, b, order = 100, args = None, Ws = None, 
        xis = None, adaptive = False, threshold = 1E-6, increment = 1, 
        verbose = False, full_output = False):
        
        """
        This function implements a Gaussian quadrature numerical integration
        scheme. It is used if the monotonicity = 'integrated rectifier', for 
        which monotonicity is ensured by integrating a strictly positive 
        function obtained from a rectifier.
        
        Variables:
            
            ===================================================================
            General variables
            ===================================================================
        
            f
                [function] : function to be integrated element-wise.
                
            a
                [float or vector] : lower bound for integration, defined as 
                either a scalar or a vector.
                
            b
                [float or vector] : upper bound for integration, defined as 
                either a scalar or a vector.
                
            order - [default = 100]
                [integer] : order of the Legendre polynomial used for the 
                integration scheme..
                
            args - [default = None]
                [None or dictionary] : a dictionary with supporting keyword
                arguments to be passed to the function.
                
            Ws - [default = None]
                [vector] : weights of the integration points, can be calculated
                in advance to speed up the computation. Is calculated by the
                integration scheme, if not specified.
                
            xis - [default = None]
                [vector] : positions of the integration points, can be 
                calculated in advance to speed up the computation. Is 
                calculated by the integration scheme, if not specified.
                
            full_output - [default = False]
                [boolean] : Flag for whether the positions and weights of the 
                integration points should returned along with the integration
                results. If True, returns a tuple with (results,order,xis,Ws).
                If False, only returns results.
                
            ===================================================================
            Adaptive integration variables
            ===================================================================
            
            adaptive - [default = False]
                [boolean] : flag which determines whether the numerical scheme
                should adjust the order of the Legendre polynomial adaptively
                (True) or use the integer provided by 'order' (False).
                
            threshold - [default = 1E-6]
                [float] : threshold for the difference in the adaptive 
                integration, adaptation stops after difference in integration 
                result falls below this value.
                
            increment - [default = 1]
                [integer] : increment by which the order is increased in each 
                adaptation cycle. Higher values correspond to larger steps.
                
            verbose - [default = False]
                [boolean] : flag which determines whether information about the
                integration process should be printer to console (True) or not
                (False).
                
        """
        
        import numpy as np
        import copy
        
        # =========================================================================
        # Here the actual magic starts
        # =========================================================================
        
        # If adaptation is desired, we must iterate; prepare a flag for this 
        repeat      = True
        iteration   = 0
        
        # Iterate, if adaptation = True; Otherwise, iteration stops after one round
        while repeat:
            
            # Increment the iteration counter
            iteration   += 1
        
            # If required, determine the weights and positions of the integration
            # points; always required if adaptation is active
            if Ws is None or xis is None or adaptive == True:
                
                # Weights and integration points are not specified; calculate them
                # To get the weights and positions of the integration points, we must
                # provide the *order*-th Legendre polynomial and its derivative
                # As a first step, get the coefficients of both functions
                coefs       = np.zeros(order+1)
                coefs[-1]   = 1
                coefs_der   = np.polynomial.legendre.legder(coefs)
                
                # With the coefficients defined, define the Legendre function
                LegendreDer = np.polynomial.legendre.Legendre(coefs_der)
                
                # Obtain the locations of the integration points
                xis = np.polynomial.legendre.legroots(coefs)
                
                # Calculate the weights of the integration points
                Ws  = 2.0/( (1.0-xis**2)*(LegendreDer(xis)**2) )
                
            # If any of the boundaries is a vector, vectorize the operation
            if not np.isscalar(a) or not np.isscalar(b):
                
                # If only one of the bounds is a scalar, vectorize it
                if np.isscalar(a) and not np.isscalar(b):
                    a       = np.ones(b.shape)*a
                if np.isscalar(b) and not np.isscalar(a):
                    b       = np.ones(a.shape)*b
                

                # Alternative approach, more amenable to dimension-sensitivity in
                # the function f. To speed up computation, pre-calculate the limit
                # differences and sum
                lim_dif = b-a
                lim_sum = b+a
                result  = np.zeros(a.shape)
                
                # print('limdifshape:'+str(lim_dif.shape))
                # print('resultshape:'+str(result.shape))
                
                # =============================================================
                # To understand what's happening here, consider the following:
                # 
                # lim_dif and lim_sum   - shape (N)
                # funcres               - shape (N) up to shape (N-by-C-by-C)
                
                # If no additional arguments were given, simply call the function
                if args is None:
                    
                    result  = lim_dif*0.5*(Ws[0]*f(lim_dif*0.5*xis[0] + lim_sum*0.5))
                    
                    for i in np.arange(1,len(Ws)):
                        result  += lim_dif*0.5*(Ws[i]*f(lim_dif*0.5*xis[i] + lim_sum*0.5))
                        
                # Otherwise, pass the arguments on as well
                else:
                    
                    funcres     = f(
                        lim_dif*0.5*xis[0] + lim_sum*0.5,
                        *args)
                    
                    # =========================================================
                    # Depending on what shape the output function returns, we 
                    # must take special precautions to ensure the product works
                    # =========================================================
                    
                    # If the function output is the same size as its input
                    if len(funcres.shape) == len(lim_dif.shape):
                        
                        result  = lim_dif*0.5*(Ws[0]*funcres)
                        
                        for i in np.arange(1,len(Ws)):
                            
                            funcres     = f(
                                lim_dif*0.5*xis[i] + lim_sum*0.5,
                                *args)
                            
                            result  += lim_dif*0.5*(Ws[i]*funcres)
                             
                    # If the function output has one dimension more than its
                    # corresponding input
                    elif len(funcres.shape) == len(lim_dif.shape)+1:
                        
                        result  = np.einsum(
                            'i,ij->ij',
                            lim_dif*0.5*Ws[0],
                            funcres)
                        
                        for i in np.arange(1,len(Ws)):
                            
                            funcres     = f(
                                lim_dif*0.5*xis[i] + lim_sum*0.5,
                                *args)
                            
                            result  += np.einsum(
                                'i,ij->ij',
                                lim_dif*0.5*Ws[i],
                                funcres)

                    # If the function output has one dimension more than its
                    # corresponding input
                    elif len(funcres.shape) == len(lim_dif.shape)+2:
                        
                        result  = np.einsum(
                            'i,ijk->ijk',
                            lim_dif*0.5*Ws[0],
                            funcres)
                        
                        for i in np.arange(1,len(Ws)):
                            
                            funcres     = f(
                                lim_dif*0.5*xis[i] + lim_sum*0.5,
                                *args)
                            
                            result  += np.einsum(
                                'i,ijk->ijk',
                                lim_dif*0.5*Ws[i],
                                funcres)
                        
                    else:
                        
                        raise Exception('Shape of input dimension is '+\
                        str(lim_sum.shape)+' and shape of output dimension is '+\
                        str(funcres.shape)+'. Currently, we have only implemented '+\
                        'situations in which input and output are the same shape, '+\
                        'or where output is one or two dimensions larger.')

            else:
                    
                # Now start the actual computation.
                
                # If no additional arguments were given, simply call the function
                if args is None:
                    result  = (b-a)*0.5*np.sum( Ws*f( (b-a)*0.5*xis+ (b+a)*0.5 ) )
                # Otherwise, pass the arguments on as well
                else:
                    result  = (b-a)*0.5*np.sum( Ws*f(
                        (b-a)*0.5*xis + (b+a)*0.5,
                        *args) )
                
            # if adaptive, store results for next iteration
            if adaptive:
                
                # In the first iteration, just store the results
                if iteration == 1:
                    previous_result = copy.copy(result)
                
                # In later iterations, check integration process
                else:
                    
                    # How much did the results change?
                    change          = np.abs(result-previous_result)
                
                    # Check if the change in results was sufficient
                    if np.max(change) < threshold or iteration > 1000:
                        
                        # Stop iterating
                        repeat      = False
                        
                        if iteration > 1000 and self.verbose:
                            print('WARNING: Adaptive integration stopped after '+\
                            '1000 iteration cycles. Final change: '+str(change))
                                
                        # Print the final change if required
                        if verbose and self.verbose:
                            print('Final maximum change of Gauss Quadrature: ' + \
                                  str(np.max(change)))
                                
                # If we must still continue repeating, increment order and store
                # current result for next iteration
                if repeat:
                    order           += increment
                    previous_result = copy.copy(result)
              
            # If no adaptation is required, simply stop iterating
            else:
                repeat  = False
            
        # If full output is desired
        if full_output:
            result  = (result,order,xis,Ws)
            
        if verbose and self.verbose:
            print('Order: '+str(order))
        
        return result
    
    def projectedNewton(self, x0, fun, args = None, method = 'trueHessian', rtol_Jdef = 1E-6, rtol_gdef = 1E-6, itmaxdef = 30, epsilon = 0.01):
        
        import copy
        import numpy as np
        import scipy
        
        def Armijo(xk, gk, pk, Jk, Ik, fun, args = None, itmax = 15, sigma = 1E-4, beta = 2):
            
            """
            xk      : current iterate
            gk      : gradient at gk
            pk      : search direction at xk
            Jxk     : objective function at xk
            J       : objective function
            Ik      : set of locally optimal coordinates on the boundary
            """
            
            import numpy as np
            
            # Iteration counter
            it  = 0
            
            Jxk_alpha_lin   = Jk + 1
            
            alpha   = beta
            
            while Jk < Jxk_alpha_lin and it < itmax:
                
                alpha           /= beta
                alpha_pk        = alpha*pk
                
                xk_alpha        = np.maximum(0, xk - alpha_pk)
                
                Jxk_alpha       = fun(xk_alpha[:,0],A=args[0],k=args[1],all_outputs=False) # Does not compute jac and hess
                alpha_pk[Ik]    = xk[Ik] - xk_alpha[Ik]
                
                Jxk_alpha_lin   = Jxk_alpha + sigma*np.dot(gk.T, alpha_pk)
                # print('jxk: '+str(Jxk_alpha_lin)+' | '+str(Jk))
                
                it              += 1
                
            if Jk < Jxk_alpha_lin:
                
                # print('Line Search reached max number of iterations.')
                
                xkh     = xk
                xkh[Ik] = 0
                index   = np.where(np.logical_and(xkh > 0, pk > 0))[0]
                # print(index)
                # print(xk)
                # print(pk)
                if len(index) == 0:
                    alpha   = 1
                else:
                    alpha   = np.min(xk[index]/pk[index])
                    
                # raise Exception
                    
            return alpha
            
        def projectGradient(xk,gk):
            
            import numpy as np
            
            Pgk     = gk
            Pgk[np.where(np.logical_and(xk == 0, gk >= 0))] = 0
            
            
            return Pgk
        
        
        def is_symmetric(a, tol=1e-8):
            return np.all(np.abs(a-a.T) < tol)
        
        rtol_J  = rtol_Jdef
        rtol_g  = rtol_gdef
        itmax   = itmaxdef
        
        # Initialize parameters
        if len(np.asarray(x0).shape) < 2:
            x0      = np.asarray(x0)[:,np.newaxis]
            # raise Exception("x0 must be a column vector.")
        if any(x0 < 0):
            raise Exception("Initial conditions must be positive.")
            
        xk              = copy.copy(x0)
        Jk, gk, Hk      = fun(xk[:,0],A=args[0],k=args[1])
        gk              = gk[:,np.newaxis]
        dim             = len(x0)
        
        norm_Pg0        = np.linalg.norm(projectGradient(xk,gk))
        norm_Pgk        = norm_Pg0
        
        tol_g           = norm_Pg0*rtol_g
        rdeltaJ         = rtol_J+1
        Jold            = Jk
        it              = 0
        
        # Start iterating
        while rdeltaJ > rtol_J and norm_Pgk > tol_g and it < itmax:
            
            # Define search direction
            wk          = np.linalg.norm(xk - np.maximum(0, xk - gk))
            
            epsk        = np.minimum(epsilon, wk)
            
            Ik          = np.where(np.logical_and(xk[:,0] <= epsk, gk[:,0] > 0))[0]
            
            if not len(Ik) == 0: # If Ik is not empty
            
                hxk         = np.diag(Hk)           # Extract the Hessian's diagonal
                zk          = np.zeros(dim)
                zk[Ik]      = copy.copy(hxk[Ik])
                Hk[Ik,:]    = 0
                Hk[:,Ik]    = 0
                Hk          += np.diag(zk)             # Buffer the Hessian's diagonal
                
            if method == 'trueHessian':
                
                Lk      = scipy.linalg.cholesky(Hk, lower = True)
                pk      = scipy.linalg.solve(
                    Lk.T,
                    scipy.linalg.solve(
                        Lk,
                        gk))
                
            elif method == 'modHessian':
                
                itmax   = 100
                
                try: 
                    
                    Lk      = scipy.linalg.cholesky(Hk, lower = True)
                    pk      = scipy.linalg.solve(
                        Lk.T,
                        scipy.linalg.solve(
                            Lk,
                            gk))
                    
                except: # Hessian is not pd
                    
                    if not is_symmetric(Hk):
                    
                        Hk  = Hk + Hk.T
                        Hk  /= 2
                        
                    print('Hessian is not pd.')
                    
                    # Eigendecomposition
                    eigval, eigvec = scipy.linalg.eig(Hk)
                    
                    eigval  = np.abs(eigval)
                    eigval  = np.maximum(eigval, 1E-8)[:,np.newaxis]
                    
                    pk      = np.dot(
                        eigvec,
                        ( (1/eigval)*np.dot(eigvec.T, gk) ) )
                        
            elif method == 'gradient': # Revert to gradient descent
            
                itmax   = 1000
                pk      = gk
                
            else:
                
                raise Exception("method not implemented yet.")
            
            # Do a line search
            alphak  = Armijo(
                xk      = xk, 
                gk      = gk, 
                pk      = pk, 
                Jk      = Jk, 
                Ik      = Ik, 
                fun     = fun, 
                args    = args)
            
            # Update
            # xk      = np.maximum(0, xk - np.dot(alphak,pk))
            xk      = np.maximum(0, xk - alphak*pk)
                        
            # Evaluate objective function
            Jk, gk, Hk = fun(xk[:,0],A=args[0],k=args[1])
            gk         = gk[:,np.newaxis]
            
            # Convergence criteria
            rdeltaJ     = np.abs(Jk - Jold)/np.abs(Jold)
            Jold        = Jk
            norm_Pgk    = np.linalg.norm(projectGradient(xk,gk))
            it          += 1
            
        # Broken out of while loop
        xopt    = copy.copy(xk)[:,0]
        
        if it < itmax:
            
            message     = 'success'
            
        else:
            
            message     = 'maxIt'
            # print('Reached max number of iterations during optimization.')
            # raise Exception
            
        return xopt
            
    
    def adaptation_cross_terms(self, increment = 1E-6, chronicle = False):
        
        """
        This function adapts a map with cross-terms.
        
        """
        
        import copy
        from scipy.optimize import minimize
            
        def cell_to_term(cell):
            
            # Create basis function
            term    = []
            for idx,order in enumerate(cell):
                term    += [idx]*order
            
            # If term is nonlinear, add a Hermite function modifier
            if self.polynomial_type.lower() == 'hermite function' and len(term) > 0:
                term    += ['HF']
            
            return term
        
        def construct_component_function(multi_index_matrix):
            
            # Find all active and proposed cells
            nonzero_cells   = np.asarray(np.where(multi_index_matrix != 0)).T
        
            # Create couters for the monotone and nonmonotone terms
            term_counter                = -1
            
            # Initiate a list for the indices of the proposed and original cells
            proposed_cells  = []
            original_cells  = []
            
            # Initiate monotone and nonmonotone lists
            monotone        = []
            nonmonotone     = []
            
            # Go through all cells
            for cell in nonzero_cells:
                
                # Increment the term counter
                term_counter    += 1
                
                # Store this index, if this term was proposed
                if self.multi_index_matrix[tuple(cell)] < 0:
                    proposed_cells      .append(term_counter)
                else:
                    original_cells      .append(term_counter)
                
                # This cell depends on the last dimension
                if cell[-1] > 0:
                    
                    term    = cell_to_term(cell)
                    
                    # Add it to the monotone term
                    monotone.append(copy.deepcopy(term))
                 
                # This cell does not depend on the last dimension
                else:
                    
                    term    = cell_to_term(cell)
                    
                    # Add it to the monotone term
                    nonmonotone.append(copy.deepcopy(term))
                    
            return monotone, nonmonotone, proposed_cells, original_cells
        
        

        # If we are chronicling the results, create a dictionary for the outputs
        if chronicle:
            chronicle_dict  = {}
        
        # So, let's adapt a map with cross-terms. Let's go through each of
        # the map component functions
        for k in range(self.D):
            
            # =============================================================
            # Initiation
            # =============================================================
            
            # If we chronicle, create a key for this map component
            if chronicle:
                chronicle_dict[k]   = {}
            
            # Create a multi-index matrix
            multi_index_matrix  = np.zeros(tuple([self.adaptation_max_order+1]*(k+1+self.skip_dimensions)),dtype=int)
            
            # The zero entry corresponds to a constant; activate it
            index   = [0]*(k+1+self.skip_dimensions)
            multi_index_matrix[tuple(index)]   = 1
            
            # One down in the last dimension corresponds to a marginal map;
            # activate that one too
            index   = [0]*(k+self.skip_dimensions)+[1]
            multi_index_matrix[tuple(index)]   = 1
            
            # Store this matrix
            self.multi_index_matrix     = multi_index_matrix
            
            # Concatenate the coefficients, and define the divisor
            coeffs  = np.asarray(list(copy.copy(self.coeffs_nonmon[k])) + list(copy.copy(self.coeffs_mon[k])))
            div     = len(self.coeffs_nonmon[k])
            
            # Minimize the objective function
            opt     = minimize(
                method  = 'BFGS',#'L-BFGS-B',
                fun     = self.objective_function,
                jac     = self.objective_function_jacobian,
                x0      = coeffs,
                args    = (k,div))
            
            # Retrieve the optimized coefficients
            coeffs          = copy.copy(opt.x)
            
            # Save the optimized coefficients
            self.coeffs_nonmon[k]   = copy.copy(coeffs[:div])
            self.coeffs_mon[k]      = copy.copy(coeffs[div:])
            
            # =============================================================
            # Begin iteration
            # =============================================================
            
            repeat      = True
            iterations  = 0
            
            # If we chronicle, store results for this iteration
            if chronicle:
                chronicle_dict[k][iterations]  = {
                    'monotone'          : copy.deepcopy(self.monotone[k]),
                    'nonmonotone'       : copy.deepcopy(self.nonmonotone[k]),
                    'coeffs_nonmon'     : copy.copy(self.coeffs_nonmon[k]),
                    'coeffs_mon'        : copy.copy(self.coeffs_mon[k]),
                    'multi_index_matrix': copy.copy(self.multi_index_matrix)}
            
            while repeat:
                
                
                
                # Increment the iteration counter
                iterations      += 1
            
                # =========================================================
                # Find all candidate cells
                # =========================================================
                
                # Find all active cells
                nonzero_cells   = np.asarray(np.where(self.multi_index_matrix > 0)).T
                
                # Go through all of these cells, check for zero-valued neighbours
                for cell in nonzero_cells:
                    
                    # Go through all dimensions of this cell
                    for idx in range(k+1+self.skip_dimensions):
                        
                        # Does the cell before exist?
                        if cell[idx] - 1 >= 0:
                            
                            # Create index before
                            index       = list(copy.copy(cell))
                            index[idx]  -= 1
                            
                            # Check multi_index_matrix
                            if self.multi_index_matrix[tuple(index)] <= 0:
                                self.multi_index_matrix[tuple(index)] -= 1
                             
                        # Does the cell after exist?
                        if cell[idx] + 1 < self.adaptation_max_order+1:
                            
                            # Create index before
                            index       = list(copy.copy(cell))
                            index[idx]  += 1
                            
                            # Check multi_index_matrix
                            if self.multi_index_matrix[tuple(index)] <= 0:
                                self.multi_index_matrix[tuple(index)] -= 1
                                
                # Are there any cells which were proposed?
                proposed_cells  = np.asarray(np.where(self.multi_index_matrix < 0)).T
                
                # If no cells have been proposed, end the while loop
                if len(proposed_cells) == 0:
                    break
                
                # Find all multi_index_matrix entries at the edges
                for cell in proposed_cells:
                    
                    # Go through all coordinate indices
                    for val in cell:
                        
                        # If this index is on the lower boundary
                        if val == 0:
                
                            # Reduce the multi_index_matrix entry further
                            self.multi_index_matrix[tuple(cell)] -= 1
                            
                # Find the reduced set of indices - only those whose count
                # is equal to the dimension of the matrix
                proposed_cells  = np.asarray(np.where(self.multi_index_matrix <= -(k+1+self.skip_dimensions))).T
                
                print(self.multi_index_matrix)
                            
                # =========================================================
                # Iterate through all proposed cells
                # =========================================================
                
                # Extract initial coefficients
                coeffs      = copy.copy(np.asarray(list(copy.copy(self.coeffs_nonmon[k])) + list(copy.copy(self.coeffs_mon[k]))))
                
                # Calculate the reference objective function
                obj_ref     = self.objective_function(
                    coeffs      = coeffs, 
                    k           = k, 
                    div         = div)
                
                # Pre-allocate space for the gradients
                grads       = np.zeros(len(proposed_cells))
                
                # Iterate through all proposed cells
                for idx,cell in enumerate(proposed_cells):
                    
                    # Reset the multi_index_matrix
                    self.multi_index_matrix[self.multi_index_matrix < 0]    = 0
                    
                    # Write in the cell
                    self.multi_index_matrix[tuple(cell)]    = -1
                
                    # Update the map component specifications
                    monotone, nonmonotone, prop, orig = construct_component_function(
                        multi_index_matrix  = self.multi_index_matrix)                    
                
                    # =====================================================
                    # Update the stored maps with the candidate components
                    # =====================================================
                    
                    # Store these map components
                    self.monotone[k]        = copy.deepcopy(monotone)
                    self.nonmonotone[k]     = copy.deepcopy(nonmonotone)
                    
                    # Update the coefficients
                    coeffs_new              = np.ones(len(nonmonotone) + len(monotone))*self.coeffs_init + increment
                    coeffs_new[orig]        = copy.copy(coeffs)
                
                    # Update the divisor
                    div                     = len(nonmonotone)
                
                    # Re-write the functions
                    self.function_constructor_alternative(k = k)
                
                    # Update the basis functions
                    self.Psi_mon[k]         = copy.copy(self.fun_mon[k](copy.copy(self.X),self))
                    self.Psi_nonmon[k]      = copy.copy(self.fun_nonmon[k](copy.copy(self.X),self))
                
                    # =====================================================
                    # Determine the gradient
                    # =====================================================
                
                    # Evaluate the gradient through finite differences
                    obj_off     = self.objective_function(
                        coeffs      = coeffs_new, 
                        k           = k, 
                        div         = div)
                    
                    # Finite difference evaluation
                    grads[idx]  = (obj_off - obj_ref)/increment
                
                                
                # =========================================================
                # Add the cell with the strongest gradient
                # =========================================================
                
                # Find the strongest gradient
                minidx  = np.where(np.abs(grads) == np.max(np.abs(grads)))[0][0]
                
                # Reset the multi_index_matrix
                self.multi_index_matrix[self.multi_index_matrix < 0]    = 0
                
                # Find the entry we want to add
                added_cell  = proposed_cells[minidx]
                
                # print(added_cell)
                
                # Set that entry to "proposed" in the multi_index_matrix
                self.multi_index_matrix[tuple(added_cell)] = -1
                
                # =========================================================
                # Construct the new map components
                # =========================================================
                
                # Find all active and proposed cells
                nonzero_cells   = np.asarray(np.where(self.multi_index_matrix != 0)).T
                
                # Update the map component specifications
                monotone, nonmonotone, prop, orig = construct_component_function(
                    multi_index_matrix  = self.multi_index_matrix)
                
                # Set that entry to "active" in the multi_index_matrix
                self.multi_index_matrix[tuple(added_cell)] = 1
                
                # =========================================================
                # Update the stored maps with the candidate components
                # =========================================================
                
                coeffs_new                  = np.ones(len(nonmonotone) + len(monotone))*self.coeffs_init
                coeffs_new[orig]            = copy.copy(coeffs)
                
                # Update the divisor
                div                         = len(nonmonotone)
                
                # Store these map components
                self.monotone[k]        = copy.deepcopy(monotone)
                self.nonmonotone[k]     = copy.deepcopy(nonmonotone)
                
                # Re-write the functions
                self.function_constructor_alternative(k = k)
            
                # Update the basis functions
                self.Psi_mon[k]         = copy.copy(self.fun_mon[k](copy.copy(self.X),self))
                self.Psi_nonmon[k]      = copy.copy(self.fun_nonmon[k](copy.copy(self.X),self))
                
                # =========================================================
                # Update the coefficients
                # =========================================================

                # # Minimize the objective function
                # opt     = minimize(
                #     method  = 'BFGS',#'L-BFGS-B',
                #     fun     = self.objective_function,
                #     jac     = self.objective_function_jacobian,
                #     x0      = coeffs_new,
                #     args    = (k,div))
                
                # Minimize the objective function
                opt     = minimize(
                    method  = 'L-BFGS-B',
                    fun     = self.objective_function,
                    x0      = coeffs_new,
                    args    = (k,div))
                
                # Retrieve the optimized coefficients
                coeffs          = copy.copy(opt.x)
                
                # Save the optimized coefficients
                self.coeffs_nonmon[k]   = copy.copy(coeffs[:div])
                self.coeffs_mon[k]      = copy.copy(coeffs[div:])
            
                # If we chronicle, store results for this iteration
                if chronicle:
                    chronicle_dict[k][iterations]  = {
                        'monotone'          : copy.deepcopy(self.monotone[k]),
                        'nonmonotone'       : copy.deepcopy(self.nonmonotone[k]),
                        'coeffs_nonmon'     : copy.copy(self.coeffs_nonmon[k]),
                        'coeffs_mon'        : copy.copy(self.coeffs_mon[k]),
                        'multi_index_matrix': copy.copy(self.multi_index_matrix)}
            
                # =========================================================
                # Should we stop?
                # =========================================================
            
                if iterations >= self.adaptation_max_iterations:
                    break
            
        # =================================================================
        # Are we done yet?
        # =================================================================
        
        if chronicle:
            
            import pickle
            
            # Pickle the adaptation dictionary
            pickle.dump(
                chronicle_dict,
                open('dictionary_adaptation_chronicle.p','wb'))
                
    
    
    #%%
    
    class rectifier:
        

        def __init__(self, mode = 'softplus', delta = 1E-8):
            
            """
            This object specifies what function is used to rectify the monotone
            map component functions if monotonicity = 'integrated rectifier',
            before the rectifier's output is integrated to yield a monotone
            map component in x_k.
            
            Variables:
            
                mode - [default = 'softplus']
                    [string] : keyword string defining which function is used
                    to rectify the map component functions.
                    
                delta - [default = 1E-8]
                    [float] : a small offset value to prevent arithmetic under-
                    flow in some of the rectifier functions.
            """
            
            self.mode       = mode
            self.delta      = delta
            
        def evaluate(self,X):
            
            """
            This function evaluates the specified rectifier.
            
            Variables:
            
                X
                    [array] : an array of function evaluates to be rectified.
            """
            
            if self.mode == 'squared':
                
                res             = X**2
                
            elif self.mode == 'exponential':
                
                res             = np.exp(X)
                
            elif self.mode == 'expneg':
                
                res             = np.exp(-X)

            elif self.mode == 'softplus':
                
                a               = np.log(2)
                aX              = a*X
                below           = (aX < 0)
                aX[below]       = 0
                res             = np.log(1 + np.exp(-np.abs(a*X))) + aX
                
            elif self.mode == 'explinearunit':
                
                res             = np.zeros(X.shape)
                res[(X < 0)]    = np.exp(X[(X < 0)])
                res[(X >= 0)]   = X[(X >= 0)]+1
                
            return res
        
        def inverse(self,X):
            
            """
            This function evaluates the inverse of the specified rectifier.
            
            Variables:
            
                X
                    [array] : an array of function evaluates to be rectified.
            """
            
            if len(np.where(X < 0)[0] > 0):
                raise Exception("Input to inverse rectifier are negative.")
            
            if self.mode == 'squared':
                
                raise Exception("Squared rectifier is not invertible.")
                
            elif self.mode == 'exponential':
                
                res             = np.log(X)
                
            elif self.mode == 'expneg':
                
                res             = -np.log(X)

            elif self.mode == 'softplus':
                
                a               = np.log(2)
                
                opt1            = np.log(np.exp(a*X) - 1)
                opt2            = X
                
                opt1idx         = (opt1-opt2 >= 0)
                opt2idx         = (opt1-opt2 < 0)
                
                res             = np.zeros(X.shape)
                res[opt1idx]    = opt1[opt1idx]
                res[opt2idx]    = opt2[opt2idx]
                
            elif self.mode == 'explinearunit':
                
                res             = np.zeros(X.shape)
                
                below           = (X < 1)
                above           = (X >= 1)
                
                res[below]      = np.log(X[below])
                res[above]      = X - 1
                
            return res
        
        def evaluate_dx(self, X):
            
            """
            This function evaluates the derivative of the specified rectifier.
            
            Variables:
            
                X
                    [array] : an array of function evaluates to be rectified.
            """
            
            if self.mode == 'squared':
                
                res             = 2*X
                
            elif self.mode == 'exponential':
                
                res             = np.exp(X)
                
            elif self.mode == 'expneg':
                
                res             = -np.exp(-X)
                
            elif self.mode == 'softplus':
                
                a               = np.log(2)
                res             = 1/(1 + np.exp(-a*X))
                
            elif self.mode == 'explinearunit':
                
                below           = (X < 0)
                above           = (X >= 0)
                
                res             = np.zeros(X.shape)
                
                res[below]      = np.exp(X[below])
                res[above]      = 0
                
            return res
                
        def evaluate_dfdc(self, f, dfdc):
            
            """
            This function evaluates terms used in the optimization of the map
            components if monotonicity = 'separable monotonicity'.
            """
            
            if self.mode == 'squared':
                
                raise Exception("Not implemented yet.")
                
            elif self.mode == 'exponential':
                
               # https://www.wolframalpha.com/input/?i=derivative+of+exp%28f%28c%29%29+wrt+c
                
                res             = np.exp(f)
                
                # Combine with dfdc
                res             = np.einsum(
                    'i,ij->ij',
                    res,
                    dfdc)
                
            elif self.mode == 'expneg':
                
                # https://www.wolframalpha.com/input/?i=derivative+of+exp%28-f%28c%29%29+wrt+c
                
                res             = -np.exp(-f)
                
                # Combine with dfdc
                res             = np.einsum(
                    'i,ij->ij',
                    res,
                    dfdc)
                
            elif self.mode == 'softplus':
                
                # https://www.wolframalpha.com/input/?i=derivative+of+log%282%5Ef%28c%29%2B1%29%2Flog%282%29+wrt+c
                
                # Calculate the first part
                a               = np.log(2)
                res             = 1/(1 + np.exp(-a*f))
                
                # Combine with dfdc
                res             = np.einsum(
                    'i,ij->ij',
                    res,
                    dfdc)

            elif self.mode == 'explinearunit':
                
                raise Exception("Not implemented yet.")
                
            return res
        
        def logevaluate(self,X):
            
            """
            This function evaluates the logarithm of the specified rectifier.
            
            Variables:
            
                X
                    [array] : an array of function evaluates to be rectified.
            """
            
            if self.mode == 'squared':
                
                res             = np.log(X**2)
                
            elif self.mode == 'exponential':
                
                # res             = np.log(np.abs(np.exp(X))) # -marked-
                # res             = X
                if self.delta == 0:
                    res             = X
                else:
                    res         = np.log(np.exp(X) + self.delta)
                
            elif self.mode == 'expneg':
                
                res             = -X

            elif self.mode == 'softplus':
                
                a               = np.log(2)
                aX              = a*X
                below           = (aX < 0)
                aX[below]       = 0
                res             = (np.log(1 + np.exp(-np.abs(a*X))) + aX)
                
                res             = np.log(res + self.delta)
                
            elif self.mode == 'explinearunit':
                
                res             = np.zeros(X.shape)
                res[(X < 0)]    = np.exp(X[(X < 0)])
                res[(X >= 0)]   = X[(X >= 0)]+1
                
                res             = np.log(res)
                
            return res

            