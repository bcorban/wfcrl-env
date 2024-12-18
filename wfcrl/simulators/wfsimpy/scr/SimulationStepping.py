# This file contains functions
# InitWFSim (initialization)
# WFSim_timestepping 

import numpy as np
import copy
import time
from wfcrl.simulators.wfsimpy.scr.SpatialDiscretization import meshing
from wfcrl.simulators.wfsimpy.scr.Making_matrices import Make_Ax_b, Computesol, MapSolution, Compute_B1_B2_bc, Solution_space


# -- INITWFSIM  Initializes the WFSim model
def InitWFSim(Wp, options, plotMesh):
    
    # Create empty structs
    sys = type("sys", (object,), {}) # This struct will contain all the system matrices at time k
    sol = type("sol", (object,), {})  # This struct will contain the solution (flowfields, power, ...) at time k

    # Import simulation scenario (meshing, atmospheric properties, turbine settings)
    Wp = meshing(Wp, plotMesh, plotMesh)

    # Initialize time vector for sol at time k = 0
    sol.k = 0
    sol.time = 0
    
    # Initialize flow fields as uniform ('no turbines present yet')
    #print(f'Initialize flow fields as uniform')
    sol.u = Wp.site.u_Inf *  np.ones((Wp.mesh.Nx,Wp.mesh.Ny))
    sol.v = Wp.site.v_Inf *  np.ones((Wp.mesh.Nx,Wp.mesh.Ny)) 
    sol.p = Wp.site.p_init *  np.ones((Wp.mesh.Nx,Wp.mesh.Ny)) 

    # Initialize the linearized solution variables, if necessary
    if options.Linearversion:
        sol.ul = sol.u
        sol.vl = sol.v
        sol.pl = sol.p
        sol.du, sol.dv, sol.dp  = [np.zeros((Wp.mesh.Nx,Wp.mesh.Ny)) for _ in range(3)]

    # Compute boundary conditions and system matrices B1, B2.
    #print(f'Compute boundary conditions and system matrices B1, B2')
    #t1 = time.time()
    sys.B1, sys.B2, sys.bc  = Compute_B1_B2_bc(Wp)
    sys.pRCM              = []    # Load empty vector
    #t2 = time.time()
    #print(f'cpu time = {t2-t1} s')
    
    # Compute projection matrices Qsp and Bsp. These are only necessary if
    # the continuity equation is projected away (2015 ACC paper, Boersma).
    if options.Projection:
        sys.Qsp, sys.Bsp  = Solution_space(sys.B1, sys.B2, sys.bc)
        Wp.Nalpha           = sys.Qsp.shape[1]
    
    return Wp,sol,sys


#-- WFSIM_TIMESTEPPING Propagates the WFSim solution one timestep forward
def WFSim_timestepping( sol, sys, Wp, turbInput, modelOptions ):
    
    # Write control settings to the solution
    if sol.k == 0:
        turbInput.dCT_prime = np.zeros((Wp.turbine.N,1))
    elif modelOptions.Linearversion:
        #turbInput.dCT_prime = turbInput.CT_prime - sol.turbInput.CT_prime
        print('Linear model not implemented yet.')

    # Import necessary information from sol_in (previous timestep)
    sol.k += 1                                  # Propagate forward in time
    sol.turbInput = turbInput   
    sol.time += Wp.sim.h                         # Propagate forward in time [s]
    # sol_out      = copy.deepcopy(sol_in)
    #sol_out = type('sol_out',(object,),{})
    #sol_out.k    = sol_in.k + 1            # Propagate forward in time
    #sol_out.time = sol_in.time + Wp.sim.h  # Propagate forward in time [s]
    sol.uk   = copy.deepcopy(sol.u)
    sol.vk   = copy.deepcopy(sol.v)
    #sol_out.u    = sol_in.u
    #sol_out.v    = sol_in.v
    #sol_out.p    = sol_in.p

    #sol_out.turbInput = turbInput   
    
    # Copy relevant system matrices from previous time
    #sys_out = type("sys_out", (), {})
    #sys_out.B1 = sys_in.B1
    #sys_out.B2 = sys_in.B2
    #sys_out.bc = sys_in.bc
    #sys_out.pRCM = sys_in.pRCM

    #if modelOptions.Projection:
        #sys_out.Qsp = sys_in.Qsp
        #sys_out.Bsp = sys_in.Bsp
    
    # Check if pRCM is properly defined
    if len(sys.pRCM) <= 0 and sol.k > 1 :
        print('pRCM not assigned. Please run [sol,sys] = WFSim_timestepping(..) \
              at sol.k = 1 to determine pRCM.')

    # Load convergence settings
    conv_eps = modelOptions.conv_eps
    if sol.k > 1:
        max_it = modelOptions.max_it_dyn
    else:
        max_it = modelOptions.max_it

    # Initialize default convergence parameters
    it         = 0
    eps        = 1e19
    epss       = 1e20
    #modelOptions.printConvergence = True

    # Convergence until satisfactory solution has been found
    while ( eps > conv_eps and it < max_it and eps < epss ) :
        it   += 1
        epss = eps

        # Create system matrices sys.A and sys.b for our nonlinear model,
        # where WFSim basically comes down to: sys.A*sol.x=sys.b.
        #print(f'Create system matrices sys.A and sys.b')
        #t1 = time.time()
        sol, sys = Make_Ax_b(Wp, sys, sol, modelOptions)
        #t2 = time.time()
        
        # Compute solution sol.x for sys.A*sol.x = sys.b.
        #print(f'Compute solution sol.x')
        #t3 = time.time()
        sol, sys = Computesol(sys, sol, it, modelOptions)
        #t4 = time.time()
        
        # Map solution sol.x to the flow field sol.u, sol.v, sol.p.
        #print(f'Map solution sol.x to the flow field')
        #t5 = time.time()
        sol, eps = MapSolution(Wp, sol, it, modelOptions)
        #t6 = time.time()

        #print(f'cpu time: {t2-t1}, {t4-t3}, {t6-t5} s')

    return sol, sys







