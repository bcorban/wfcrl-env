#  This file contains functions
#  solverSet_default
#  layoutSet_sowfa_9turb_apc_alm_turbl
#  controlSet_sowfa_9turb_apc_alm_turbl

import numpy as np
import scipy.io as sio
import os


## -- layouts
def layoutSet_sowfa(timestep=1.0, Drotor=126.4,  
                    Nx=50, Ny=25, 
                    turb_pos_x = np.array([0.4, 1.2820]), Lx = 2132.0,
                    turb_pos_y = np.array([0.4, 0.3976982]), Ly = 800.0,
                    u_Inf = 8.0641,  v_Inf = 0, p_init = 0.0,
                    powerscale=0.95, forcescale=1.4,
                    lm_slope=1.2, d_lower=73.3, d_upper=601.9,
                    description='2 NREL 5MW turbines case, turbulent inflow, based on a SOWFA ALM simulation where turbine 1 is yawed'):
    Wp = type("Wp", (object,), {})
    Wp.description = description

    Wp.sim = type("sim", (object,), {})
    Wp.sim.h = timestep          # timestep (s)
    Wp.sim.startUniform = True   # Start from a uniform flow field (T) 
                                 # or from a fully developed waked flow field (F).

    Wp.turbine = type("turbine", (object,), {})
    # X-coordinates of turbines (m)                             
    Wp.turbine.Crx = turb_pos_x*1e3  
    # Y-coordinates of turbines (m) 
    Wp.turbine.Cry = turb_pos_y*1e3  
    # Rotor diameter (m), note that WFSim only supports a uniform Drotor for now
    Wp.turbine.Drotor = Drotor
    # Turbine power scaling  (to be tuned)
    Wp.turbine.powerscale = powerscale
    # Turbine force scaling  (to be tuned)
    Wp.turbine.forcescale = forcescale

    Wp.site = type("site", (object,), {})
    Wp.site.u_Inf = u_Inf     # Initial long. wind speed in m/s
    Wp.site.v_Inf = v_Inf       # Initial lat. wind speed in m/s
    Wp.site.p_init = p_init       # Initial values for pressure terms (Pa)
    Wp.site.lm_slope = lm_slope     # Mixing length in x-direction (m)   (to be tuned)
    Wp.site.d_lower = d_lower     # Turbulence model gridding property (to be tuned)
    Wp.site.d_upper = d_upper    # Turbulence model gridding property (to be tuned)
    Wp.site.Rho = 1.20         # Air density

    Wp.mesh = type("mesh", (object,), {})
    Wp.mesh.Lx = Lx           # Domain length in x-direction
    Wp.mesh.Ly = Ly           # Domain length in y-direction
    Wp.mesh.Nx = Nx           # Number of cells in x-direction
    Wp.mesh.Ny = Ny           # Number of cells in y-direction

    return Wp




# -- get controls from registed sowfa data
def controlSet_sowfa(Wp, loadFileName='DB_sowfa_9turb_apc_alm_turbl.mat'):
    turbInputSet = type("turbInputSet", (object,), {})

    les_db_path = './controlDefinitions/LES_database'
    loadedDB = sio.loadmat(os.path.join(les_db_path, loadFileName))

    n_data = loadedDB['turbInput']['t'].shape[1]
    n_turbine = loadedDB['turbInput']['phi'][0][0].shape[0]
    turbInputSet.t = np.zeros(n_data)
    turbInputSet.phi = np.zeros((n_turbine,n_data))
    turbInputSet.CT_prime = np.zeros((n_turbine,n_data))
    for i in range(n_data):
        turbInputSet.t[i] = loadedDB['turbInput']['t'][0][i].squeeze()
        turbInputSet.phi[:,i] = loadedDB['turbInput']['phi'][0][i].squeeze()
        turbInputSet.CT_prime[:,i] = loadedDB['turbInput']['CT_prime'][0][i].squeeze()

    turbInputSet.interpMethod = 'linear'; # Linear interpolation over time
    
    if len(Wp.turbine.Crx) != n_turbine:
        print('Number of turbines in layout does not match your controlSet.')

    return turbInputSet


# -- This is the default solver set for open-loop simulations
def solverSet(Wp):
    modelOptions = type("modelOptions", (object,), {})

    # -- Model settings (recommended: leave default)
    # Solve WFSim by projecting away the continuity equation (bool). Default: false.
    modelOptions.Projection        = False         
    # Calculate linear system matrices of WFSim (bool).              Default: false.
    modelOptions.Linearversion     = False       
     # Calculate linear solution of WFSim (bool).                    Default: false.  
    modelOptions.exportLinearSol   = False        
    # Calculate pressure fields. Default: '~scriptOptions.Projection'
    modelOptions.exportPressures   = not modelOptions.Projection     

    # -- Convergence settings (recommended: leave default)
    # Print convergence values every timestep.                       Default: false.
    modelOptions.printConvergence = False
    # Convergence threshold. Default: 1e-6.
    modelOptions.conv_eps         = 1e-6
    # Maximum number of iterations for k > 1. Default: 1.
    modelOptions.max_it_dyn       = 1   

    if Wp.sim.startUniform==1 :
        # Maximum n.o. of iterations for k == 1, when startUniform = 1.
        modelOptions.max_it = 1             
    else :
        # Maximum n.o. of iterations for k == 1, when startUniform = 0.
        modelOptions.max_it = 50              

    return modelOptions


def displaySet():
    verboseOptions = type("verboseOptions", (object,), {})
    verboseOptions.printProgress = True    # Print progress in cmd window every timestep. Default: true.
    verboseOptions.Animate       = False   # Plot flow fields every [X] iterations (0: no plots). Default: 10.
    verboseOptions.plotMesh      = False   # Plot mesh, turbine locations, and print grid offset values. Default: false.
    return verboseOptions



def layoutSet_sowfa_9turb_apc_alm_turbl(Nx=100, Ny=42):
    Wp = type("Wp", (object,), {})
    Wp.description = '9 NREL 5MW turbines case, based on a SOWFA ALM simulation'

    Wp.sim = type("sim", (object,), {})
    Wp.sim.h = 1.0               # timestep (s)
    Wp.sim.startUniform = True   # Start from a uniform flow field (T) 
                                 # or from a fully developed waked flow field (F).

    Wp.turbine = type("turbine", (object,), {})
    # X-coordinates of turbines (m)                             
    Wp.turbine.Crx = np.array([0.4048,0.4024,0.40,1.0368,1.0344,1.0320,1.6688,1.6663,1.6639])*1e3
    # Y-coordinates of turbines (m)
    Wp.turbine.Cry = np.array([1.1584,0.7792,0.40,1.1543,0.7752,0.3960,1.1503,0.7711,0.3919])*1e3
    # Rotor diameter (m), note that WFSim only supports a uniform Drotor for now
    Wp.turbine.Drotor = 126.4
    # Turbine power scaling  (to be tuned)
    Wp.turbine.powerscale = 0.99  
    # Turbine force scaling  (to be tuned)
    Wp.turbine.forcescale = 1.9

    Wp.site = type("site", (object,), {})
    Wp.site.u_Inf = 12.0214    # Initial long. wind speed in m/s
    Wp.site.v_Inf = 0.0        # Initial lat. wind speed in m/s
    Wp.site.p_init = 0.0       # Initial values for pressure terms (Pa)
    Wp.site.lm_slope = 0.05    # Mixing length in x-direction (m)   (to be tuned)
    Wp.site.d_lower = 140.0    # Turbulence model gridding property (to be tuned)
    Wp.site.d_upper = 1000.0   # Turbulence model gridding property (to be tuned)
    Wp.site.Rho = 1.20         # Air density

    Wp.mesh = type("mesh", (object,), {})
    Wp.mesh.Lx = 2518.8        # Domain length in x-direction
    Wp.mesh.Ly = 1558.4        # Domain length in y-direction
    Wp.mesh.Nx = Nx           # Number of cells in x-direction
    Wp.mesh.Ny = Ny            # Number of cells in y-direction


    # Tuning notes 'apc_9turb_alm_turb' (Sep 11th, 2017):
    # Ranges: lmu= 0.1:0.1:2.0, f = 0.8:0.1:2.5, m = 1:8, n = 1:4

    # Tuning notes on Nov 5, 2018
    # Optimal out of:
    #     'forcescale', 1.5:0.1:2.5, ...
    #     'lm_slope', 0.005:0.005:0.10, ...
    #     'd_lower', 0.1:20:200.1,...
    #     'd_upper', 300:50:1000,...

    return Wp

