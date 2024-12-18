# This file contains functions:
#   adjust_turbine_grids
#   Dynamical
#   Actuator
#   BoundaryConditions
#   Updateboundaries 
#   read_maps
import os
import numpy as np
from scipy.sparse import spdiags, bmat, csr_matrix, diags, bmat, block_diag

from scipy.interpolate import  interp1d

def read_maps():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    cpct_data = np.genfromtxt(os.path.join(parent_dir, 'cartos','cpct_data.dat'))
    # cpct_data = np.genfromtxt(os.path.join(current_dir, 'cartos','cpct_data.dat'))
    w  = np.array(cpct_data[:,0])
    cp = np.array(cpct_data[:,1])
    # Saturation de ct
    map_ct = np.array(cpct_data[:,2])
    map_ct[map_ct>0.999] = 0.999
    ct = map_ct
    
    w_to_ct = interp1d(w, ct, fill_value=(ct[0],ct[-1]))
    w_to_cp = interp1d(w, cp, fill_value=(cp[0],cp[-1]))
    ct_to_cp = interp1d(ct, cp, fill_value=(cp[0],cp[-1]))
    
    '''
    plt.plot(w,cp,label='cp')
    plt.plot(w,ct,label='ct')
    plt.grid()
    plt.legend()
    plt.show()
    '''
    
    return w_to_ct, w_to_cp, ct_to_cp

    
    

def adjust_turbine_grids(Wp, stepk,  alpha=0):
     # Calculate location of turbines in grid and grid mismatch
    yline = []
    xline = []
    ylinev = []
    ldx = Wp.mesh.ldxx[:,0]
    ldy = Wp.mesh.ldyy[0,:]
    ldy2 = Wp.mesh.ldyy2[0,:]
    Drotor = Wp.turbine.Drotor
    alpha = alpha/180*np.pi # transform to arc
    n_turbines = len(Wp.turbine.Crx)

    turb_lines = []
    for i in range(n_turbines):
        # Calculate cells relevant for turbine (x-dir) on primary grid
        
        x_left = np.argmin(np.abs(ldx - (Wp.turbine.Crx[i] - Drotor/2*np.sin(alpha[i]) ))) 
        x_right = np.argmin(np.abs(ldx - (Wp.turbine.Crx[i] + Drotor/2*np.sin(alpha[i]) ))) 
        if x_left <= x_right:
            xline_i = np.arange(x_left, x_right+1, 1).astype(np.int32)
        # else:
        #     xline_i = np.arange(x_left, x_right+1, -1).astype(np.int32)
        else:
            xline_i = np.arange(x_left, x_right - 1, -1).astype(np.int32)
        # print(f'alpha : {alpha[i]}, x left : {x_left}, x right : {x_right}')
        xline.append(xline_i)
        
        #  Calculate cells closest to turbines (y-dir) on both grids
        ML_prim = np.argmin(np.abs(ldy - (Wp.turbine.Cry[i] - Drotor/2*np.cos(alpha[i]))))
        MR_prim = np.argmin(np.abs(ldy - (Wp.turbine.Cry[i] + Drotor/2*np.cos(alpha[i]))))

        yline_i = np.arange(MR_prim+1, ML_prim,  -1).astype(np.int32)
        ylinev_i = np.arange(MR_prim + 2 ,ML_prim, -1).astype(np.int32)
        yline.append(yline_i)          # turbine cells for primary grid
        ylinev.append(ylinev_i)       # turbine cells for secondary grid


        #if len(xline_i)*ratio < len(yline_i):
        #    xline_i = np.arange(x_left, x_right+2).astype(np.int32)

        #print(f'ratio={ratio}, Drotor={Drotor}, {alpha[i]}')
        #print(f'{x_left},{x_right}')
        #print(f'{ML_prim},{MR_prim}')

        if len(yline_i) >= len(xline_i) :
            ratio = int(len(yline_i)/len(xline_i))
            rest = len(yline_i) % len(xline_i)
            turb_lines_i = np.zeros((len(yline_i)+1,4)).astype(np.int32)
            if rest == 0:
                turb_lines_i[:-1,0] = np.repeat(xline_i.reshape((len(xline_i),1)), ratio, axis=0).squeeze()
            elif rest >=2 :
                xline_i_adjust = np.repeat(xline_i.reshape((len(xline_i),1)), ratio+1, axis=0).squeeze()
                turb_lines_i[:-1,0] = xline_i_adjust[int(rest/2):int(rest/2)+len(yline_i)]
            else:
                xline_i_adjust = np.repeat(xline_i.reshape((len(xline_i),1)), ratio+1, axis=0).squeeze()
                turb_lines_i[:-1,0] = xline_i_adjust[rest:rest+len(yline_i)]

            turb_lines_i[:-1,1] = yline_i
            xlinev_i = np.hstack((turb_lines_i[:-1,0], xline_i[-1]))
            turb_lines_i[:,2] = xlinev_i
            turb_lines_i[:,3] = ylinev_i
        else:

            ratio = int(len(xline_i)/len(yline_i))
            rest = len(xline_i) % len(yline_i) 
            turb_lines_i = np.zeros((len(xline_i)+1,4)).astype(np.int32)

            turb_lines_i[:-1,0] = xline_i
            if rest == 0:
                turb_lines_i[:-1,1] =  np.repeat(yline_i.reshape((len(yline_i),1)), ratio, axis=0).squeeze()
            elif rest >=2:
                yline_i_adjust = np.repeat(yline_i.reshape((len(yline_i),1)), ratio+1, axis=0).squeeze()
                turb_lines_i[:-1,1] = yline_i_adjust[int(rest/2):int(rest/2)+len(xline_i)]
            else:
                yline_i_adjust = np.repeat(yline_i.reshape((len(yline_i),1)), ratio+1, axis=0).squeeze()
                turb_lines_i[:-1,1] = yline_i_adjust[rest:rest+len(xline_i)]

            xlinev_i = np.hstack((turb_lines_i[:-1,0], xline_i[-1]))
            turb_lines_i[:,2] = xlinev_i
            turb_lines_i[:,3] = np.hstack((turb_lines_i[:-1,1], yline_i_adjust[-1]))

        turb_lines.append(turb_lines_i)
    
        #if (stepk == 1 or stepk == 105 or stepk == 205) and i==0 :
        #    print(f'turb elements: {turb_lines_i}, alpha = {alpha[i]}')
        #print(f'turb elements: {turb_lines_i}, alpha = {alpha[i]}')
        
    Wp.mesh.turb_lines = turb_lines
    #if (stepk == 1 or stepk == 105 or stepk == 205)  :
    #    print(f'turb elements: {Wp.mesh.turb_lines[0]}')

    #Wp.mesh.xline = xline
    #Wp.mesh.yline = yline
    #Wp.mesh.ylinev = ylinev

    return Wp, turb_lines


# -- compute actuator forces
def Actuator(Wp, sol, options):
    # Import variables
    Nx              = Wp.mesh.Nx
    Ny              = Wp.mesh.Ny
    dyy2            = Wp.mesh.dyy2
    #xline           = Wp.mesh.xline    # this can be enhanced with yaw
    #yline           = Wp.mesh.yline    # this can be enhanced
    #ylinev          = Wp.mesh.ylinev
    Rho             = Wp.site.Rho
    Drotor          = Wp.turbine.Drotor
    powerscale      = Wp.turbine.powerscale
    N               = Wp.turbine.N
    F               = Wp.turbine.forcescale
    input           = sol.turbInput
    Projection      = options.Projection
    Linearversion   = options.Linearversion

    Ar              = np.pi*(0.5*Drotor)**2

    # create outputs
    Sm = type("Sm", (object,), {})
    dSm = type("dSm", (object,), {})
    output = type("output", (object,), {})

    # Input x-mom nonlinear and linear
    Sm.x, Sm.dx    = [csr_matrix((Nx-3, Ny-2),dtype=np.float32) for _ in range(2)]
    # Input y-mom nonlinear and linear            
    Sm.y, Sm.dy    = [csr_matrix((Nx-2, Ny-3),dtype=np.float32) for _ in range(2)]       
    # Input x-mom nonlinear and linear qlpv
    Sm.xx, Sm.dxx  = [csr_matrix(((Nx-3)*(Ny-2), 2*N),dtype=np.float32) for _ in range(2)]     
    # Input y-mom nonlinear and linear qlpv
    Sm.yy, Sm.dyy  = [csr_matrix(((Nx-2)*(Ny-3), 2*N),dtype=np.float32) for _ in range(2)]     

    if Linearversion :
        Smdu  = csr_matrix((Nx-3,Ny-2))
        Smdv  = csr_matrix((Nx-2,Ny-3))

    U, Ue = [], []
    phi, meanUe, gamma, dCT, Power, CT = [np.zeros(N) for _ in range(6)]
    CP = np.zeros(N)
    if options.saveForce: Forces = np.zeros((N,2))

    for kk in range(N):
        phi[kk] = np.arctan(sol.v[0, 0] / sol.u[0, 0])    # angle between wind direction and x-axis
        if options.control_yaw:
            gamma[kk] = input.phi[kk]                     # yaw angle [degree]
        else:
            gamma[kk] = 0                                 # if yaw is not controlled, set to zero
    
    #if sol.k ==0 or sol.k == 105 or sol.k == 205:
    #    print(f'step= {sol.k}, \n yaw = {gamma}\n, phi = {phi}')

    #t1 = time.time()
    # print(gamma)
    Wp, turb_lines = adjust_turbine_grids(Wp, sol.k, alpha=phi/np.pi*180+gamma)
    #t2 = time.time()
    #print(f'cpu find place = {t2-t1} s')
    #print(f'xline = {Wp.mesh.xline}')
    #print(f'yline {Wp.mesh.yline}')

    for kk in range(N):
        
        #x  = Wp.mesh.xline[kk]    # Turbine x-pos in field
        #y  = Wp.mesh.yline[kk]    # Turbine y-pos in field
        #yv = Wp.mesh.ylinev[kk]   # Corrected turbine y-pos in field
        turb_lines_i = Wp.mesh.turb_lines[kk] 
        x = turb_lines_i[:-1,0]
        y = turb_lines_i[:-1,1]
        xv =  turb_lines_i[:,2]
        yv =  turb_lines_i[:,3]

        #print(f'alpha={phi+gamma}')
        #print(f'turb_lines_{kk}={turb_lines_i}')

        vv = 0.5 * np.diff(sol.v[xv, yv]) + sol.v[xv[:-1], yv[:-1]]
        uu = sol.u[x, y]

        U.append(np.sqrt(uu**2 + vv**2))                     # wind speed
        
        Ue.append(np.cos(gamma[kk] / 180 * np.pi) * U[kk])   # effective wind spped wrt turbine kk
        meanUe = np.mean(Ue[kk])

        if options.control_pitch:
            CT[kk] = input.CT_prime[kk]                       # Import CT_prime from inputData
        else:
            CT[kk] = Wp.w_to_ct(meanUe)
            
        ## Thrust force   
        CT_prime =   F*CT[kk]    
        Fthrust =  0.5 * Rho * Ue[kk]**2 * CT_prime            
        Fx = Fthrust * np.cos(phi[kk] + gamma[kk] * np.pi / 180)
        #Fy = Fthrust * np.sin(phi[kk] + gamma[kk] * np.pi / 180)
        Fy = -Fthrust * np.sin(phi[kk] + gamma[kk] * np.pi / 180)
        #print(f'turb{kk}, F={Fthrust}, Fx={Fx}, Fy={Fy}')
        
        Forces[kk,:] = np.array([Fx.mean(), Fy.mean()])

        ## Power
        if options.use_maps_for_cp and not options.control_pitch:
            CP_kk = powerscale*Wp.w_to_cp(meanUe)
        elif options.control_pitch:
            CP_kk = powerscale*Wp.ct_to_cp(CT[kk])
        else:
            CP_kk = powerscale*CT[kk]
        CP[kk] = CP_kk
        
        #print(f'meanUe={meanUe}, CT_kk={CT[kk]}, CP_kk={CP_kk}, at time={sol.time}')
            
        Power[kk] =  0.5 * Rho * Ar * CP_kk * np.mean(Ue[kk]**3)
        
        #Power[kk] = powerscale * 0.5 * Rho * Ar * CT[kk] * np.mean(Ue[kk])**3
        #if sol.k ==0 or sol.k == 105 or sol.k == 205:
        #    print(f'turb {kk} Ue_mean={np.mean(Ue[kk])}' )
        
        ## Input to Ax=b
        Sm.x[x-2, y-1] = - Fx.T * dyy2[0, y-1]           # Input x-mom nonlinear
        Sm.y[x[1:]-1, y[1:]-2] = Fy[1:].T * dyy2[0, y[1:]]   # Input y-mom nonlinear

        # Apply the force to the trailing cells to achieve a higher (LES-like) wake deflection
        Sm.y[x[1:], y[1:]-2] = Fy[1:].T * dyy2[0, y[1:]]
        Sm.y[x[1:]+1, y[1:]-2] = Fy[1:].T * dyy2[0, y[1:]]

        
        # Matrices for linear version
        if Linearversion :
            dCT[kk] = input.dCT_prime[kk]
            
            dFthrustdCT = F * 0.5 * Rho * Ue[kk]**2
            dFxdCT = dFthrustdCT * np.cos(phi[kk] + gamma[kk] * np.pi / 180)
            dFydCT = dFthrustdCT * np.sin(phi[kk] + gamma[kk] * np.pi / 180)

            Sm.dx[x-2, y-1] = -dFxdCT.T * dCT[kk] * dyy2[0, y-1].T
            Sm.dy[x-1, y[1:]-2] = dFydCT[1:].T * dCT[kk] * dyy2[0, y[1:]].T

            dFdu = F * Rho * np.cos(gamma[kk] * np.pi / 180)**2 * CT[kk] * uu
            dFdv = F * Rho * np.cos(gamma[kk] * np.pi / 180)**2 * CT[kk] * vv
            Smdu[x-2, y-1] = -dFdu.T * dyy2[0, y-1].T
            Smdv[x-1, y-2] = dFdv.T * dyy2[0, y-2].T

            Smdu_diag = diags(Smdu.toarray().T.flatten(order='F'))  # Transposition et mise en forme du vecteur
            Smdv_diag = diags(Smdv.toarray().T.flatten(order='F'))  # Idem pour Smdv
            zero_block = csr_matrix(((Ny - 2) * (Nx - 2), (Ny - 2) * (Nx - 2)))


            dSm.dx = block_diag((Smdu_diag, Smdv_diag, zero_block))
            
            # following for projection
            tempdx = csr_matrix((Nx-3, Ny-2))
            tempdy = csr_matrix((Nx-2, Ny-3))
            tempdx[x-2, y-1] = -dFxdCT.T * dyy2[0, y-1].T
            Sm.dxx[:, kk] = tempdx.toarray().flatten(order='F')  # Input matrix (beta) x-mom linear       
            tempdy[x-1, y[1:]-2] = dFydCT[1:].T * dyy2[0, y[1:]].T
            Sm.dyy[:, kk] = tempdy.toarray().flatten(order='F')  # Input (beta) y-mom linear qlpv
        
        if Projection:
            ## Input to qLPV
            # Clear for multiple turbine case                                  
            tempx = csr_matrix((Nx-3, Ny-2))
            tempy = csr_matrix((Nx-2, Ny-3))
            tempx[x-2, y-1] = -Fx.T * dyy2[0, y-1].T
            Sm.xx[:, kk] = tempx.toarray().flatten(order='F') / CT[kk]
            Sm.xx[:, N + kk] = tempx.toarray().flatten(order='F')                              

            if gamma[kk] != 0:
                Sm.xx[:, kk] /= 2
                Sm.xx[:, N + kk] /= (2 * gamma[kk])  # Input x-mom nonlinear qlpv

            tempy[x-1, y[1:]-2] = Fy[1:].T * dyy2[0, y[1:]].T
            Sm.yy[:, kk] = tempy.toarray().flatten(order='F') / CT[kk]
            Sm.yy[:, N + kk] = tempy.toarray().flatten(order='F')

            if gamma[kk] != 0:
                Sm.yy[:, kk] /= 2
                Sm.yy[:, N + kk] /= (2 * gamma[kk])  # Input y-mom nonlinear qlpv

    ## Write to outputs
    sol.turbine = type("solturbine", (object,), {})

    if options.savePower: sol.turbine.power = Power
    if options.saveCP: sol.turbine.CT = CT
    if options.saveCT: sol.turbine.CP = CP
    if options.saveForce: 
        sol.turbine.Fx = Forces[:,0]
        sol.turbine.Fy = Forces[:,1]
 
    output.Sm  = Sm   # Sm contains sparse csr_matrix matrix
    if Linearversion > 0 :
        output.dSm = dSm

    return output, sol





def Dynamical(Wp, StrucDiscretization, sol, dt, Linearversion):
    Rho  = Wp.site.Rho
    dxx  = Wp.mesh.dxx
    dyy  = Wp.mesh.dyy
    dxx2 = Wp.mesh.dxx2
    dyy2 = Wp.mesh.dyy2
    Nx   = Wp.mesh.Nx
    Ny   = Wp.mesh.Ny
    u    = sol.uk
    v    = sol.vk

    StrucDynamical = type("StrucDynamical", (object,), {})
    
    # Fully implicit (page 248 Versteeg) See also page 257
    StrucDiscretization.ax.aP += Rho*dxx*dyy2/dt     # Rho.*dxx.*dyy2./dt = a_P^0
    StrucDiscretization.ay.aP += Rho*dxx2*dyy/dt

    StrucDynamical.ccx    = (Rho*dxx[2:-1,1:-1].T * dyy2[2:-1,1:-1].T /dt).flatten(order='F')
    StrucDynamical.cx     = StrucDynamical.ccx * u[2:-1,1:-1].T.flatten(order='F')

    StrucDynamical.ccy    = (Rho*dxx2[1:-1,2:-1].T * dyy[1:-1,2:-1].T /dt).flatten(order='F')
    StrucDynamical.cy     = StrucDynamical.ccy * (v[1:-1,2:-1].T).flatten(order='F')

    if Linearversion:
        StrucDiscretization.dax.P   += Rho*dxx*dyy2/dt     
        StrucDiscretization.day.P   += Rho*dxx2*dyy/dt

        # Création des matrices diagonales
        ccx_diag = spdiags(StrucDynamical.ccx, 0, len(StrucDynamical.ccx), len(StrucDynamical.ccx))
        ccy_diag = spdiags(StrucDynamical.ccy, 0, len(StrucDynamical.ccy), len(StrucDynamical.ccy))
        # Création d'une matrice creuse vide
        empty_matrix = csr_matrix(((Ny - 2) * (Nx - 2), (Ny - 2) * (Nx - 2)))

        # Construction de la matrice bloc-diagonale
        StrucDynamical.dcdx = bmat([[ccx_diag, None],
                                    [None, ccy_diag],
                                    [None, empty_matrix]])

    return StrucDiscretization,StrucDynamical


def BoundaryConditions(Wp, StrucDiscretization, sol, Linearversion):
    # Import variables
    Nx = Wp.mesh.Nx
    Ny = Wp.mesh.Ny
    u  = sol.u
    v  = sol.v

    StrucBCs = type("StrucBCs", (object,), {})

    # Execute script
    ax = StrucDiscretization.ax
    ay = StrucDiscretization.ay

    # Créer des matrices creuses
    bbx = csr_matrix(((Nx-3)*(Ny-2), (Nx-3)*(Ny-2) + (Nx-2)*(Ny-3) + (Nx-2)*(Ny-2)), dtype=np.float32)
    bby = csr_matrix(((Nx-2)*(Ny-3), (Nx-3)*(Ny-2) + (Nx-2)*(Ny-3) + (Nx-2)*(Ny-2)), dtype=np.float32)

    # Zero gradient outflow pour la direction u
    ax.aP[Nx-1, 1:Ny-1] -= ax.aE[Nx-1, 1:Ny-1]  # NORTH
    ax.aP[0:Nx-1, Ny-1] -= ax.aN[0:Nx-1, Ny-1]  # EAST
    ax.aP[0:Nx-1, 1] -= ax.aS[0:Nx-1, 1]

    if Linearversion:
        dax = StrucDiscretization.dax
        day = StrucDiscretization.day
        
        dax.P[Nx-1, 1:Ny-1] -= dax.E[Nx-1, 1:Ny-1]  # NORTH
        dax.P[0:Nx-1, Ny-1] -= dax.N[0:Nx-1, Ny-1]
        dax.P[0:Nx-1, 1] -= dax.S[0:Nx-1, 1]

        # Conditions d'entrée (pas de gradient nul)
        dax.P[1:3, :] += dax.aW[1:3, 0:Ny] * u[0:2, 0:Ny]  # y momentum côté ouest

        # Gradient nul
        dax.NW[:, 1] -= dax.NW[:, 0]  # + dax.aPN[:, 1] * u[:, 1]
        dax.NE[0:Nx, 1] -= dax.NE[0:Nx, 0]
        dax.SW[:, -2] -= dax.SW[:, -1]  # - dax.aS[0:Nx, -2] * u[0:Nx, -2]
        dax.SE[0:Nx, -2] -= dax.SE[0:Nx, -1]  # - dax.aS[0:Nx, -2] * u[0:Nx, -2]
        day.SW[-2, :] -= day.SW[-1, :]  # day.aW[-2, :] * v[-2, :]
        day.NW[-2, :] -= day.NW[-1, :]  # day.aW[-2, :] * v[-2, :]

        # dax.NW[1:3, 0:Ny-1] -= dax.aS[1:3, 0:Ny-1] * u[1:3, 1:Ny]

    # Dérivées de gradient nul supplémentaires
    # dax.NW[0:Nx-1, 1] += dax.aS[0:Nx-1, 1] * u[1:Nx, 1]
    # dax.NE[0:Nx-1, 1] += dax.aS[0:Nx-1, 1] * u[1:Nx, 1]

    # Pour la direction v
    ay.aP[Nx-1, 0:Ny] -= ay.aE[Nx-1, 0:Ny]
    ay.aP[0:Nx, Ny-1] -= ay.aN[0:Nx, Ny-1]
    ay.aP[0:Nx, 2] -= ay.aS[0:Nx, 2]  # changé à 3 3 2 au lieu de 2 2 1

    if Linearversion:
        day.P[Nx-1, 0:Ny] -= day.E[Nx-1, 0:Ny]
        day.P[0:Nx, Ny-1] -= day.N[0:Nx, Ny-1]
        day.P[0:Nx, 2] -= day.S[0:Nx, 2]  # changé à 3 3 2 au lieu de 2 2 1

    # Conditions d'entrée pour le modèle non linéaire
    # bx   = kron([1;zeros(Nx-4,1)],(ax.aW(3,2:end-1).*u(2,2:end-1))'); %changed to 3: 2 instead of 2:2
    bx = np.kron(np.array([1] + [0]*(Nx-4)), (ax.aW[2, 1:Ny-1] * u[1, 1:Ny-1]).T)
    by = np.concatenate([v[0, 2:Ny-1] * ay.aW[1, 2:Ny-1], np.zeros((Nx-3)*(Ny-3))])

    if Linearversion:
        # Conditions d'entrée pour le modèle linéaire
        bbx[0:Ny-2, 0:Ny-2] = diags((dax.aW[2, 1:Ny-1] * u[1, 1:Ny-1]).T)
        bby[0:Ny-2, 0:Ny-2] = 0 * diags((day.aW[1, 1:Ny-1] * v[0, 1:Ny-1]).T)

        # Écriture dans la sortie
        StrucBCs.dbcdx = csr_matrix(np.vstack([bbx.toarray(), bby.toarray(), 
                                               csr_matrix((Nx-2)*(Ny-2), 
                                               (Nx-3)*(Ny-2) + (Nx-2)*(Ny-3) + (Nx-2)*(Ny-2)).toarray()]))
        StrucDiscretization.dax = dax
        StrucDiscretization.day = day

    # Écriture des sorties non linéaires
    StrucDiscretization.ax = ax
    StrucDiscretization.ay = ay
    StrucBCs.bx = bx
    StrucBCs.by = by

    return StrucDiscretization, StrucBCs


def Updateboundaries(Nx,Ny,u,v,p):
    # Three zero gradient boundary conditions
    u[:, 0] = u[:, 1]              
    u[:, Ny - 1] = u[:, Ny - 2]    # u_{i,1}  = u_{i+1,2}   for i = 1,..Nx
    u[Nx - 1, :] = u[Nx - 2, :]    # u_{Nx,J} = u_{Nx-1,J}  for J = 1,..Ny (hence u_Inf comes via first row in field)

    v[:, 1] = v[:, 2]
    v[:, 0] = v[:, 1]              # v_{I,1}  = v_{I,2}    for I = 1,..Nx
    v[:, Ny - 1] = v[:, Ny - 2]    # v_{I,Ny} = v_{I,Ny-1} for I = 1,..Nx
    v[Nx - 1, :] = v[Nx - 2, :]    # v_{Nx,j} = v_{Nx,j}   for j = 1,..Ny (hence v_Inf comes via row in field)

    p[:, 1] = p[:, 2]              # Trick to make pressure field nice
    p[:, Ny - 2] = p[:, Ny - 3]    # Trick to make pressure field nice
    p[Nx - 2, :] = p[Nx - 3, :]    # Trick to make pressure field nice
    p[:, 0] = p[:, 1]              # p_{i,1}  = p_{i+1,2}   for i = 1,..Nx
    p[:, Ny - 1] = p[:, Ny - 2]    # p_{i,Ny} = p_{i,Ny-1}  for i = 1,..Nx
    p[Nx - 1, :] = p[Nx - 2, :]    # p_{Nx,J} = p_{Nx-1,J}  for J = 1,..Ny

    # p = p + (rand(size(p))-.5)/.5*1e-2;
    # v = v + (rand(size(u))-.5)/.5*5e-1;
    return u,v,p




