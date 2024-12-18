import numpy as np
from scipy.interpolate import interp1d
import time
import pandas as pd
import os

'''
from farmSettings import *
from SimulationStepping import InitWFSim, WFSim_timestepping
'''


if __name__ == '__main__':
    #import pathlib
    #lib_path = pathlib.Path().resolve()
    #exec(open("D:/wfcrl/WFSim/wfsimpy/scr/farmSettings.py").read())
    exec(open("./scr/SpatialDiscretization.py").read())
    exec(open("./scr/SystemDescription.py").read())
    exec(open("./scr/farmSettings.py").read())
    exec(open("./scr/Making_matrices.py").read())
    exec(open("./scr/SimulationStepping.py").read())
    exec(open("./scr/PostProcessing.py").read())

    
    # timestep (s)
    h = 3.0               
    # Rotor diameter (m), note that WFSim only supports a uniform Drotor for now
    Drotor = 126.4
    # X-coordinates of turbines (km)                             
    Crx = np.array([0.4] + [0.4+4*Drotor/1e3] + [0.4+2*4*Drotor/1e3])
    # Y-coordinates of turbines (km)
    Cry = np.array([0.4]*3)

    # Turbine power scaling (to be tuned)
    powerscale = 2.3
    # Turbine force scaling (to be tuned)
    forcescale = 2.0

    u_Inf = 8.0        # Initial long. wind speed in m/s
    v_Inf = 0.0        # Initial lat. wind speed in m/s
    lm_slope = 0.03    # Mixing length in x-direction (m)   (to be tuned) 1.2
    d_lower = 190      # Turbulence model gridding property (to be tuned) 73.3
    d_upper = 1000     # Turbulence model gridding property (to be tuned) 601.9

    Lx = (Crx.max() + 4*Drotor/1e3 ) * 1e3       # Domain length in x-direction
    Ly = (Cry.max() + 0.4 ) * 1e3                # Domain length in y-direction
    dx = 10
    dy = 10
    Nx = int(Lx/dx)            # Number of cells in x-direction
    Ny = int(Ly/dy)            # Number of cells in y-direction


        
        
    # -- create scenario to simulate. 
    Wp = layoutSet_sowfa(timestep=h, Nx=Nx, Ny=Ny, Drotor=Drotor,
                         turb_pos_x = Crx,  Lx = Lx,
                         turb_pos_y = Cry,  Ly = Ly,
                         u_Inf = u_Inf, v_Inf = v_Inf,
                         powerscale=powerscale, forcescale=forcescale,
                         lm_slope=lm_slope, d_lower=d_lower, d_upper=d_upper)
    

        
    # -- Choose model solver options
    modelOptions = solverSet(Wp) 
    # control options
    modelOptions.use_maps_for_cp = True   # if true, and not pitch control, find cp from LUT(wind_speed,cp) : 
                                          # see function Actuator in SystemDescription.py 
    modelOptions.control_pitch = False    # if true, we have to decide input "turbInput.CT_prime", and cp is decided from LUT(ct,cp)
    modelOptions.control_yaw = True       # if true, we have to decide input "turbInput.phi" 
    modelOptions.max_it_dyn = 2
    # quantities to save during the simulation
    modelOptions.savePower = True  # save powers
    modelOptions.saveCP=True       # save cp
    modelOptions.saveCT=True       # save ct
    modelOptions.saveForce=True    # save forces Fx and Fy

    # -- display settings
    verboseOptions = displaySet()

    # -- Initialize WFSim model 
    Wp, sol, sys = InitWFSim(Wp, modelOptions, verboseOptions.plotMesh)
    n_turbines = Wp.turbine.N
    
    # -- load aerodynamic parameter maps
    Wp.w_to_ct, Wp.w_to_cp, Wp.ct_to_cp = read_maps()
    
    
    # -- create outputs saving directory
    res_file = f'./wfsim_res/{n_turbines}turbines_dx{dx}dy{dy}_fs{forcescale}_ps{powerscale}_lm{lm_slope}_d{d_lower}_{d_upper}/'
    if not os.path.exists(res_file):
        os.makedirs(res_file)
        print(res_file)
    # -- decide series of yaws to test
    # yaws_series = [np.array([0, 0, 0]),
    #                np.array([10, 0, 0]),
    #                np.array([20, 0, 0]),
    #                np.array([30, 0, 0]),
    #                np.array([35, 0, 0]),
    #                np.array([35, 10, 0]),
    #                np.array([35, 20, 0]),
    #                np.array([35, 30, 0]),
    #                np.array([38, 30, 0])]
    yaws_series = [np.array([38, 30, 0])]
    #
    steplen = 100                  # step numbers of each yaw in the series
    savefig_freq = 100             # figure plot and save frequency [iteration]
    NN = steplen*len(yaws_series)  # total step numbers

    Timesteps = np.zeros((NN))
    Powers, CPs, CTs, Fx, Fy = [np.zeros((NN,n_turbines)) for _ in range(5)]
    CPUTimes = np.zeros(NN)
    while sol.k < NN :

        # decide control inputs
        turbInput = type("turbInput", (object,), {}) 
        #turbInput.CT_prime = np.array([0.76]*n_turbines)
        turbInput.phi      = yaws_series[np.fix(sol.k/steplen).astype(np.int32)]


        t_start = time.time()
        # Propagate the WFSim model : forward timestep: x_k+1 = f(x_k)
        sol, sys  = WFSim_timestepping(sol, sys, Wp, turbInput, modelOptions) 
        t_end = time.time()

        CPUTimes[sol.k-1] = round(t_end - t_start, 6)
        Timesteps[sol.k-1] = sol.time
        if modelOptions.savePower: Powers[sol.k-1, :] = sol.turbine.power[:]      
        if modelOptions.saveCP: CPs[sol.k-1, :] = sol.turbine.CP[:]   
        if modelOptions.saveCT: CTs[sol.k-1, :] = sol.turbine.CT[:]  
        if modelOptions.saveForce: 
            Fx[sol.k-1, :] = sol.turbine.Fx[:]   
            Fy[sol.k-1, :] = sol.turbine.Fy[:]   
        
        if verboseOptions.printProgress :
            print(f'Simulated t({sol.k}) = {sol.time} s. ')
            #print(f'CPU: {round(t_end - t_start, 5)} s.')
            sol.Timesteps = Timesteps
            sol.Powers = Powers
            sol.CPs = CPs
            sol.CTs = CTs
            sol.Fx = Fx
            sol.Fy = Fy
            sol.CPUTimes = np.array(CPUTimes)

            if sol.k % savefig_freq == 0:  
                animation_turb(Wp, sol, n_turbines=n_turbines, idx_0=steplen-10, res_file=res_file)
                
    animation_turb(Wp, sol, n_turbines=n_turbines, idx_0=steplen-10, res_file=res_file)

    # -- save res to csv
    sol.res_file = res_file
    sol.steplen = steplen
    sol.series_test_n = len(yaws_series)
    sol.series_test_yaws = yaws_series
    save_res(sol)    
    



    