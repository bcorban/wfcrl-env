import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


# -- visualization of wind field 
def animation_turb(Wp, sol, n_turbines = 6, idx_0=300, res_file='./'):

    ldyy   = Wp.mesh.ldyy
    ldxx2  = Wp.mesh.ldxx2
    #n_turbines = Wp.turbine.N
    Drotor = Wp.turbine.Drotor

    u_Inf = Wp.site.u_Inf  # Exemple de valeur pour u_Inf, à adapter à votre contexte
    
    min_u = np.min(sol.u)
    max_u = np.max(sol.u)
    levels = np.arange(min_u*0.9, max_u* 1.1, 0.1)

    plt.figure(1)
    plt.clf()
    # -- velocity field u
    plt.subplot(2,1,1)
    cs = plt.contourf(ldxx2[:,0],ldyy[0,:],
                        sol.u.T, levels=levels, cmap='hot')
    #cs.cmap.set_over('red')
    #cs.cmap.set_under('blue')
    #cs.changed()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('u m/s')
    plt.tight_layout()
    
    # Ajuster la plage de valeurs sur la colorbar (similaire à caxis en MATLAB)
    #plt.clim(min_u - 2, u_Inf * 1.2)
    plt.colorbar(cs) # Ajuster la colorbar
    #plt.show()

    Cry    = Wp.turbine.Cry
    Crx    = Wp.turbine.Crx
    gamma = sol.turbInput.phi*np.pi/180   # Yaw angles
    #turb_coord = .5*Wp.turbine.Drotor*np.exp(1j*sol.turbInput.phi*np.pi/180)  # Yaw angles
    for kk in range(n_turbines):
        #Qx = np.linspace(Crx[kk] - np.imag(turb_coord[kk]), Crx[kk] + np.imag(turb_coord[kk]), len(Qy))    
        #Qy = np.arange(Cry[kk] - np.real(turb_coord[kk]), Cry[kk] + np.real(turb_coord[kk]) + 1, 1)
        Qx = np.linspace(Crx[kk] - 0.5*Drotor*np.sin(gamma[kk]), Crx[kk] + 0.5*Drotor*np.sin(gamma[kk]), 10)
        Qy = np.linspace(Cry[kk] + 0.5*Drotor*np.cos(gamma[kk]), Cry[kk] - 0.5*Drotor*np.cos(gamma[kk]), len(Qx))
        plt.plot(Qx, Qy, 'k')
    
    # -- velocity field v
    plt.subplot(2,1,2)
    cs = plt.contourf(ldxx2[:,0],ldyy[0,:], sol.v.T)
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    cs.changed()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.colorbar()
    plt.title('v m/s')
    plt.tight_layout()
    
    plt.savefig(f'{res_file}windfield_t={sol.time}.png')

    #Cry    = Wp.turbine.Cry
    #Crx    = Wp.turbine.Crx
    #turb_coord = .5*Wp.turbine.Drotor*np.exp(1j*sol.turbInput.phi*np.pi/180)  # Yaw angles
    for kk in range(n_turbines): 
        #Qy = np.arange(Cry[kk] - np.real(turb_coord[kk]), Cry[kk] + np.real(turb_coord[kk]) + 1, 1)
        #Qx = np.linspace(Crx[kk] - np.imag(turb_coord[kk]), Crx[kk] + np.imag(turb_coord[kk]), len(Qy))
        Qx = np.linspace(Crx[kk] - 0.5*Drotor*np.sin(gamma[kk]), Crx[kk] + 0.5*Drotor*np.sin(gamma[kk]), 1)
        Qy = np.linspace(Cry[kk] + 0.5*Drotor*np.cos(gamma[kk]), Cry[kk] - 0.5*Drotor*np.cos(gamma[kk]), len(Qx))
        plt.plot(Qx, Qy, 'k')
    #plt.show()

    # Wake mean centreline first turbine
    yline = Wp.mesh.yline
    xline = Wp.mesh.xline
    ldyy = Wp.mesh.ldyy
    dx = ldxx2[:,0][1]-ldxx2[:,0][0]

    if n_turbines == 3:
        n_affiche = 1
    if n_turbines ==6:
        n_affiche = np.min([2,n_turbines])
    if n_turbines ==9:
        n_affiche = np.min([3,n_turbines])
                 
    plt.figure(2)
    plt.clf()
    plt.subplot(2,2,1)
    for i in range(n_affiche):
        up  = np.mean(sol.u[:,yline[i]],axis=1)
        plt.plot(ldxx2[:,0], up, label=f'turb {i+1}')
    plt.axvline(x = Crx[0], ls = '-.', color = 'r')
    
    if n_turbines ==3:
        plt.axvline(x = Crx[1], ls = '-.', color = 'g')
        plt.axvline(x = Crx[2], ls = '-.', color = 'b')
    if n_turbines == 6:
        plt.axvline(x = Crx[2], ls = '-.', color = 'g')
        plt.axvline(x = Crx[4], ls = '-.', color = 'b')
    if n_turbines == 9:
        plt.axvline(x = Crx[3], ls = '-.', color = 'g')
        plt.axvline(x = Crx[6], ls = '-.', color = 'b')   
             
    plt.xlabel('x [m]')
    plt.legend()
    plt.grid()
    plt.title(f'Wake mean centreline turbine')
    plt.tight_layout()
    
    plt.subplot(2,2,2)
    for i in range(n_affiche):
        up  = sol.u[xline[i],:]
        plt.plot(ldyy[0,:], up, label=f'turb {i+1}')
        plt.xlabel('y [m]')
    plt.legend()
    plt.grid()
    plt.title(f'speed at turbines')
    plt.tight_layout()
    
    d1 = (Drotor/dx).astype(np.int16)
    nd = 1
    plt.subplot(2,2,3)
    for i in range(n_affiche):
        up  = sol.u[xline[i] + nd*d1,:]
        plt.plot(ldyy[0,:], up, label=f'turb {i+1}')
        plt.xlabel('y [m]')
    plt.legend()
    plt.grid() 
    plt.title(f'{nd}D speed at turbines')
    plt.tight_layout()
    
    plt.subplot(2,2,4)
    nd = 4
    for i in range(n_affiche):
        up  = sol.u[xline[i] + nd*d1,:]
        plt.plot(ldyy[0,:], up, label=f'turb {i+1}')
        plt.xlabel('y [m]')
    plt.legend()
    plt.grid()
    plt.title(f'{nd}D speed at turbines')
    plt.tight_layout()
    plt.savefig(f'{res_file}speed_lines_t={sol.time}.png')


    idx_0 = 300
    plt.figure(3)
    plt.clf()

    plt.subplot(2,1,1)
    for i in range(n_turbines):
        plt.plot(sol.Timesteps[idx_0:sol.k], sol.Powers[idx_0:sol.k,i]/1e3, label=f'turb {i+1}')
    plt.legend()
    plt.grid()
    plt.xlabel('time [s]')
    plt.ylabel('power [kW]')
    plt.tight_layout()

    plt.subplot(2,1,2)
    plt.plot(sol.Timesteps[idx_0:sol.k], np.sum(sol.Powers[idx_0:sol.k,:],axis=1)/1e3, label = 'total')
    plt.legend()
    plt.grid()
    plt.xlabel('time [s]')
    plt.ylabel('total power [kW]')
    plt.tight_layout()

    plt.savefig(f'{res_file}powers.png')

    # plt.show()
    #
    # if n_turbines == 3:
    #     plt.figure(4)
    #     plt.clf()
    #     for i in range(n_turbines):
    #         plt.plot(sol.Timesteps[idx_0:sol.k], sol.CPs[idx_0:sol.k,i], label=f'CP {i+1}')
    #         plt.plot(sol.Timesteps[idx_0:sol.k], sol.CTs[idx_0:sol.k,i], label=f'CT {i+1}')
    #         plt.legend()
    #     plt.grid()
    #     plt.xlabel('time [s]')
    #     plt.ylabel('CP and CT')
    #     plt.tight_layout()
    #
    # if n_turbines == 6:
    #     plt.figure(4)
    #     plt.clf()
    #     plt.subplot(2,1,1)
    #     for i in [0,2,4]:
    #         plt.plot(sol.Timesteps[idx_0:sol.k], sol.CPs[idx_0:sol.k,i], label=f'CP {i+1}')
    #         plt.plot(sol.Timesteps[idx_0:sol.k], sol.CTs[idx_0:sol.k,i], label=f'CT {i+1}')
    #         plt.legend()
    #     plt.grid()
    #     plt.xlabel('time [s]')
    #     plt.ylabel('CP and CT')
    #
    #     plt.subplot(2,1,2)
    #     for i in [1,3,5]:
    #         plt.plot(sol.Timesteps[idx_0:sol.k], sol.CPs[idx_0:sol.k,i], label=f'CP {i+1}')
    #         plt.plot(sol.Timesteps[idx_0:sol.k], sol.CTs[idx_0:sol.k,i], label=f'CT {i+1}')
    #         plt.legend()
    #     plt.grid()
    #     plt.xlabel('time [s]')
    #     plt.ylabel('CP and CT')
    #     plt.tight_layout()
    #
    # if n_turbines == 9:
    #     plt.figure(4)
    #     plt.clf()
    #     plt.subplot(3,1,1)
    #     for i in [0,3,6]:
    #         plt.plot(sol.Timesteps[idx_0:sol.k], sol.CPs[idx_0:sol.k,i], label=f'CP {i+1}')
    #         plt.plot(sol.Timesteps[idx_0:sol.k], sol.CTs[idx_0:sol.k,i], label=f'CT {i+1}')
    #         plt.legend()
    #     plt.grid()
    #     plt.xlabel('time [s]')
    #     plt.ylabel('CP and CT')
    #
    #     plt.subplot(3,1,2)
    #     for i in [1,4,7]:
    #         plt.plot(sol.Timesteps[idx_0:sol.k], sol.CPs[idx_0:sol.k,i], label=f'CP {i+1}')
    #         plt.plot(sol.Timesteps[idx_0:sol.k], sol.CTs[idx_0:sol.k,i], label=f'CT {i+1}')
    #         plt.legend()
    #     plt.grid()
    #     plt.xlabel('time [s]')
    #     plt.ylabel('CP and CT')
    #
    #     plt.subplot(3,1,3)
    #     for i in [2,5,8]:
    #         plt.plot(sol.Timesteps[idx_0:sol.k], sol.CPs[idx_0:sol.k,i], label=f'CP {i+1}')
    #         plt.plot(sol.Timesteps[idx_0:sol.k], sol.CTs[idx_0:sol.k,i], label=f'CT {i+1}')
    #         plt.legend()
    #     plt.grid()
    #     plt.xlabel('time [s]')
    #     plt.ylabel('CP and CT')
    #
    #     plt.tight_layout()
    #
    # plt.savefig(f'{res_file}CpCt.png')
    #


# -- save results
def save_res(sol, saveCPUtimes = True):

    data = {'time': sol.Timesteps}
    for i in range(n_turbines):
        data[f'power_{i}_kw'] = np.round(sol.Powers[:,i]/1e3, 2)
        data[f'CP_{i}'] = sol.CPs[:,i]
        data[f'CT_{i}'] = sol.CTs[:,i]
        data[f'Fx_{i}'] = sol.Fx[:,i]
        data[f'Fy_{i}'] = sol.Fy[:,i]
        
    data[f'total_power_kw'] = np.round(np.sum(sol.Powers,axis=1)/1e3, 2)
    
    if saveCPUtimes:
        data[f'CPUTimes'] = sol.CPUTimes
    
    res_file = sol.res_file
    df = pd.DataFrame(data)
    df.to_csv(f'{res_file}output.csv', index=False)
    
    
    # synthesis
    Powers = sol.Powers
    steplen = sol.steplen
    P0_ref = 5.7*1e6
    P0 = np.sum(Powers[steplen-5])
    ratio_p = P0/P0_ref
    plt.figure()
    plt.plot(Powers)
    plt.plot(np.sum(Powers,axis=1))
    plt.show()
    
    Powers_normalized = Powers/ratio_p
    Pk_normalized_mw = np.zeros(sol.series_test_n)
    for k in range(sol.series_test_n):
        Pk_normalized_mw[k] =np.sum(Powers_normalized[steplen*k-5])/1e6
    dP_normalized_mw = np.diff(Pk_normalized_mw)
    
    idx_max_power = np.argmax(Pk_normalized_mw)
    bestyaw = sol.series_test_yaws[idx_max_power]

    print(f'best yaw: {bestyaw}')
    print(f'powers mw : {Powers[idx_max_power].sum()}, with ratio:{ratio_p}')
    print(f'delta powers mw (normalised): {dP_normalized_mw}')
    print(f'Average CPU time: {np.mean(np.array(sol.CPUTimes))} s')
    
    with open(f'{res_file}output_synthesis.txt', 'w') as f:
        f.write(f'powers mw: {Powers/1e6}\n')
        f.write(f'powers mw (normalised to fastfarm res): {Pk_normalized_mw}, with ratio:{ratio_p}\n')
        f.write(f'delta powers mw (normalised): {dP_normalized_mw}\n')
        f.write(f'best yaw: {bestyaw}\n')
        f.write(f'Average CPU time: {np.mean(np.array(sol.CPUTimes))} s\n')
        f.close()
        
    
        
    