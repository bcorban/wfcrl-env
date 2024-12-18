# This file contains function:
#   meshing (à traduire): called in InitWFSim
#   diskfilter
#   Lmu_2D_WF
#   ConstructLmu
#   Turbulence
#   SpatialDiscr_Hybrid

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import convolve2d

def meshing( Wp, plotMesh=False, PrintGridMismatch=False ):

    # Meshing and settings function for the WFSim code
    # This code calculates the meshing and prepares the Wp struct of a
    # specific wind farm case for WFSim simulations

    # Default settings
    if Wp is None:
        raise ValueError('Please specify a meshing case.')


    # linear gridding (primary grid)
    ldx = np.linspace(0, Wp.mesh.Lx, Wp.mesh.Nx, dtype=np.float32)
    ldy = np.linspace(0, Wp.mesh.Ly, Wp.mesh.Ny, dtype=np.float32)
    
    ldxx = np.tile(ldx[:, np.newaxis], (1, Wp.mesh.Ny))  # matrix of shape (Nx,Ny)
    ldyy = np.tile(ldy, (Wp.mesh.Nx, 1))                 # matrix of shape (Nx,Ny)

    # Create secondary grid from primary grid (midpoints of primary grid)
    ldx2 = 0.5 * (ldx[:-1] + ldx[1:])                # midpoints of ldx
    ldx2 = np.append(ldx2, 2 * ldx2[-1] - ldx2[-2])  # add last point

    ldy2 = 0.5 * (ldy[:-1] + ldy[1:])                # midpoints of ldy
    ldy2 = np.append(ldy2, 2 * ldy2[-1] - ldy2[-2])  # add last point

    ldxx2 = np.tile(ldx2[:, np.newaxis], (1, Wp.mesh.Ny))   # matrix of shape (Nx,Ny)
    ldyy2 = np.tile(ldy2, (Wp.mesh.Nx, 1))                  # matrix of shape (Nx,Ny)

    # Calculate cell dimensions
    dx = np.diff(ldx)      # dx of primary grid
    dxx = np.tile(np.append(dx, dx[-1]), (Wp.mesh.Ny, 1)).T   # matrix of shape (Nx,Ny)
    dy = np.diff(ldy)      # dy of primary grid
    dyy = np.tile(np.append(dy, dy[-1]), (Wp.mesh.Nx, 1))     # matrix of shape (Nx,Ny)

    dx2 = np.diff(ldx2)    # dx of secondary grid
    dxx2 = np.tile(np.append(dx2, dx2[-1]), (Wp.mesh.Ny, 1)).T  # matrix of shape (Nx,Ny)
    dy2 = np.diff(ldy2)    # dy of secondary grid
    dyy2 = np.tile(np.append(dy2, dy2[-1]), (Wp.mesh.Nx, 1))    # matrix of shape (Nx,Ny)

    # Calculate location of turbines in grid and grid mismatch
    yline = []
    xline = []
    ylinev = []
    n_turbines = len(Wp.turbine.Crx)
    for i in range(n_turbines):
        # Calculate cells relevant for turbine (x-dir) on primary grid
        xline_i = np.argmin(np.abs(ldx - Wp.turbine.Crx[i]))  # automatically picks earliest entry in vector
        xline.append(xline_i)
        
        #  Calculate cells closest to turbines (y-dir) on both grids
        ML_prim = np.argmin(np.abs(ldy - (Wp.turbine.Cry[i] - Wp.turbine.Drotor / 2)))
        ML_sec = np.argmin(np.abs(ldy2 - (Wp.turbine.Cry[i] - Wp.turbine.Drotor / 2)))
        MR_prim = np.argmin(np.abs(ldy - (Wp.turbine.Cry[i] + Wp.turbine.Drotor / 2)))
        MR_sec = np.argmin(np.abs(ldy2 - (Wp.turbine.Cry[i] + Wp.turbine.Drotor / 2)))

        yline.append(np.arange(ML_prim, MR_prim+1))          # turbine cells for primary grid
        ylinev.append(np.arange(ML_prim, MR_prim + 2))       # turbine cells for secondary grid

        if PrintGridMismatch:
            # Calculate turbine-grid mismatch
            print(f'TURBINE {i+1} GRID MISMATCH:')
            print('                    Primary           Secondary')
            print(f'       center:   ({min(abs(Wp.turbine.Crx[i] - ldx)):.2f},  {min(abs(Wp.turbine.Cry[i] - ldy)):.2f}) m. '
                  f'({min(abs(Wp.turbine.Crx[i] - ldx2)):.2f},  {min(abs(Wp.turbine.Cry[i] - ldy2)):.2f}) m.')
            print(f'   left blade:   ({min(abs(Wp.turbine.Crx[i] - ldx)):.2f}, {ML_prim:.2f}) m. '
                  f'({min(abs(Wp.turbine.Crx[i] - ldx2)):.2f}, {ML_sec:.2f}) m.')
            print(f'  right blade:   ({min(abs(Wp.turbine.Crx[i] - ldx)):.2f}, {MR_prim:.2f}) m. '
                  f'({min(abs(Wp.turbine.Crx[i] - ldx2)):.2f}, {MR_sec:.2f}) m.')
            print(' ')


    ## Display results
    if plotMesh:
        plt.figure()
        '''
         Z1 = -2 * np.ones(ldxx.shape)
        ax = plt.axes(projection='3d')
        surf = ax.plot_surface(ldyy, ldxx, Z1, 
                       linewidth=0, antialiased=False, label='Primary mesh')
        Z2 = -1 * np.ones(ldxx2.shape)
        surf = ax.plot_surface(ldyy2, ldxx2, Z2, 
                       linewidth=0, antialiased=False, label='Secondary mesh')
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d
        '''

        for j in range(n_turbines):
            plt.plot([Wp.turbine.Cry[j] - Wp.turbine.Drotor / 2,
                       Wp.turbine.Cry[j] + Wp.turbine.Drotor / 2],
                     [Wp.turbine.Crx[j], Wp.turbine.Crx[j]],
                     linewidth=3.0, label=f'Turbine {j+1}')

        #plt.axis('equal')    
        plt.xlim([-0.1 * Wp.mesh.Ly, 1.2 * Wp.mesh.Ly])
        plt.ylim([-0.1 * Wp.mesh.Lx, 1.1 * Wp.mesh.Lx])
        plt.xlabel('y (m)')
        plt.ylabel('x (m)')
        plt.legend()
        plt.show()

    ## Export to Wp and input
    Wp.Nu = (Wp.mesh.Nx-3)*(Wp.mesh.Ny-2)   # Number of u velocities in state vector
    Wp.Nv = (Wp.mesh.Nx-2)*(Wp.mesh.Ny-3)   # Number of v velocities in state vector
    Wp.Np = (Wp.mesh.Nx-2)*(Wp.mesh.Ny-2)-2 # Number of pressure terms in state vector
    Wp.turbine.N =n_turbines

    # Write meshing
    Wp.mesh.ldxx = ldxx
    Wp.mesh.ldyy = ldyy
    Wp.mesh.ldxx2= ldxx2
    Wp.mesh.ldyy2= ldyy2
    Wp.mesh.dxx = dxx
    Wp.mesh.dyy = dyy
    Wp.mesh.dxx2 = dxx2
    Wp.mesh.dyy2 = dyy2
    Wp.mesh.xline = xline
    Wp.mesh.yline = yline
    Wp.mesh.ylinev = ylinev

    return Wp 






def diskfilter(p2):

    # Check the number of input arguments.
    rad   = p2
    crad  = np.ceil(rad-0.5)
    crad = int(crad)

    x, y = np.meshgrid(np.arange(-crad, crad + 1), np.arange(-crad, crad + 1))
    maxxy = np.maximum(np.abs(x), np.abs(y))
    minxy = np.minimum(np.abs(x), np.abs(y))

    m1 = (rad**2 < (maxxy + 0.5)**2 + (minxy - 0.5)**2) * (minxy - 0.5) + \
         (rad**2 >= (maxxy + 0.5)**2 + (minxy - 0.5)**2) * np.sqrt(np.fmax(0,rad**2 - (maxxy + 0.5)**2))
    
    m2 = (rad**2 > (maxxy - 0.5)**2 + (minxy + 0.5)**2) * (minxy + 0.5) + \
         (rad**2 <= (maxxy - 0.5)**2 + (minxy + 0.5)**2) * np.sqrt(np.fmax(0,rad**2 - (maxxy - 0.5)**2))
   

    sgrid = (rad**2 * (0.5 * (np.arcsin(m2 / rad) - np.arcsin(m1 / rad)) + 
                       0.25 * (np.sin(2 * np.arcsin(m2 / rad)) - np.sin(2 * np.arcsin(m1 / rad)))) - 
                       (maxxy - 0.5) * (m2 - m1) + (m1 - minxy + 0.5)) * \
             (((rad**2 < (maxxy + 0.5)**2 + (minxy + 0.5)**2) & 
               (rad**2 > (maxxy - 0.5)**2 + (minxy - 0.5)**2)) | 
              ((minxy == 0) & (maxxy - 0.5 < rad) & (maxxy + 0.5 >= rad)))

    sgrid += (maxxy + 0.5)**2 + (minxy + 0.5)**2 < rad**2
    sgrid[crad, crad] = min(np.pi * rad**2, np.pi / 2)

    if crad > 0 and rad > crad - 0.5 and rad**2 < (crad - 0.5)**2 + 0.25:
        m1 = np.sqrt(rad**2 - (crad - 0.5)**2)
        m1n = m1 / rad
        sg0 = 2 * (rad**2 * (0.5 * np.arcsin(m1n) + 0.25 * np.sin(2 * np.arcsin(m1n))) - m1 * (crad - 0.5))

        sgrid[2 * crad, crad] = sg0
        sgrid[crad, 2 * crad] = sg0
        sgrid[crad, 0] = sg0
        sgrid[0, crad] = sg0
        sgrid[2 * crad - 1, crad] -= sg0
        sgrid[crad, 2 * crad - 1] -= sg0
        sgrid[crad, 1] -= sg0
        sgrid[1, crad] -= sg0
    
    sgrid[crad, crad] = min(sgrid[crad, crad], 1)
    h = sgrid / np.sum(sgrid)

    return h

'''
# Exemple d'utilisation
crad = 10  # Exemple de valeur pour crad
rad = 5    # Exemple de valeur pour rad
result = compute_sgrid(crad, rad)
print(result)
'''


# 2D Lmu profile for single turbine case at (x,y)=(0,0)
#    this is the classical profile with sharp corners
def  Lmu_2D_WF(x, y, D, d_lower,d_upper,lm_slope):
    lm       = np.zeros(x.shape)
    indx = np.multiply( (x > d_lower*np.ones(x.shape)), (x < d_upper*np.ones(x.shape)))
    indx = np.multiply(indx, (y <= D/2*np.ones(y.shape)))
    indx = np.multiply(indx, (y > -D/2*np.ones(y.shape)))

    lm[indx] = (x[indx] - d_lower) * lm_slope
    return lm


def filter2(b, x, shape='same'):
    '''
    %FILTER2 Two-dimensional digital filter.
    %   Y = FILTER2(B,X) filters the data in X with the 2-D FIR
    %   filter in the matrix B.  The result, Y, is computed 
    %   using 2-D correlation and is the same size as X. 
    %
    %   Y = FILTER2(B,X,SHAPE) returns Y computed via 2-D
    %   correlation with size specified by SHAPE:
    %     'same'  - (default) returns the central part of the 
    %               correlation that is the same size as X.
    %     'valid' - returns only those parts of the correlation
    %               that are computed without the zero-padded
    %               edges, size(Y) < size(X).
    %     'full'  - returns the full 2-D correlation, 
    %               size(Y) > size(X).
    %
    %   FILTER2 uses CONV2 to do most of the work.  2-D correlation
    %   is related to 2-D convolution by a 180 degree rotation of the
    %   filter matrix.
    %
    %   Class support for inputs B,X:
    %      float: double, single
    %
    %   See also FILTER, CONV2.

    %   Copyright 1984-2022 The MathWorks, Inc. 
    '''

    stencil = np.rot90(b, 2)
    if stencil.ndim == 1 or stencil.size > x.size:

        # The filter is bigger than the input.  This is a nontypical
        # case, and it may be counterproductive to check the
        # separability of the stencil.
        y = convolve2d(x, stencil, mode=shape)

    else:
        separable = False

        # Stencil is considered to be not separable if the following test fails
        # for any reason, like non-finite stencil, empty stencil, etc.
        try:
            # Test de la séparabilité (utilisation de la décomposition SVD)
            u, s, v = np.linalg.svd(stencil, full_matrices=False)
            if s[1] <= len(stencil) * np.finfo(float).eps * s[0]:
                separable = True
        except Exception as e:
            # En cas d'erreur (ex : matrice non définie), on la traite comme non séparée
            separable = False

        if separable:
            # Cas d'un filtre séparable
            hcol = u[:, 0] * np.sqrt(s[0])
            hrow = np.conj(v[:, 0]) * np.sqrt(s[0])
            y = convolve2d(x, hcol[:, np.newaxis], mode=shape)
            y = convolve2d(y, hrow[np.newaxis, :], mode=shape)
            
            # Si l'entrée et le filtre sont tous les deux entiers, arrondir le résultat
            if np.all(np.round(stencil) == stencil) and np.all(np.round(x) == x):
                y = np.round(y)
        else:
            # Cas d'un filtre non séparable
            y = convolve2d(x, stencil, mode=shape)

    return y


def ConstructLmu(x_IF, y_IF, WD, xTurbs, yTurbs, D, d_lower, d_upper, lm_slope):
    # Check inputs
    if d_upper <= d_lower :
        print('Make sure your upper bound is larger than your lower bound on the Lmu turbulence model.')
    
    lm = np.zeros(x_IF.shape, dtype=np.float32)

    # Rotation vector
    rotWD = np.array( [[np.cos(WD), -np.sin(WD)], [np.sin(WD), np.cos(WD)]])

    # Add mixing length to the field for every turbine
    for iT in range(len(xTurbs)):
        xy_WF = np.matmul(np.array([x_IF.reshape(-1) - xTurbs[iT], y_IF.reshape(-1) - yTurbs[iT]]).T , rotWD)
        x_WF  = np.reshape(xy_WF[:,0], x_IF.shape)
        y_WF  = np.reshape(xy_WF[:,1], y_IF.shape)

        # Determine turbine-added mixing length
        lm += Lmu_2D_WF(x_WF, y_WF, D, d_lower, d_upper, lm_slope)
       
    H     = diskfilter(1)
    lm_    = filter2(H, lm)  # equivalent of filter2
    
    # % Plot results
    # clf; surf(X,Y,lm);
    # axis equal tight
    # xlabel('x (m)');
    # ylabel('y (m)');
    # title('Lmu plot (m)');
    # drawnow()

    return lm_






def Turbulence(Wp, sol, ax, ay, dax, day, Linearversion):

    xline  = Wp.mesh.xline
    yline  = Wp.mesh.yline
    Drotor = Wp.turbine.Drotor
    N      = Wp.turbine.N

    Nx     = Wp.mesh.Nx
    Ny     = Wp.mesh.Ny
    Rho    = Wp.site.Rho
    dxx    = Wp.mesh.dxx
    dyy    = Wp.mesh.dyy

    u      = sol.u
    v      = sol.v

    # Determine mixing length distribution in the field
    mixing_length = ConstructLmu(Wp.mesh.ldxx2,
                                 Wp.mesh.ldyy,
                                 np.tan(Wp.site.v_Inf/Wp.site.u_Inf),  # freestreem wind direction wrt x axis
                                 Wp.turbine.Crx,
                                 Wp.turbine.Cry,
                                 Wp.turbine.Drotor,
                                 Wp.site.d_lower,
                                 Wp.site.d_upper,
                                 Wp.site.lm_slope)

    # figure; surf(Wp.mesh.ldyy2,Wp.mesh.ldxx,mixing_length); axis equal; xlabel('y'); ylabel('x');

    # include turbulence model in equations
    # For u-momentum equation
    ax.Tnx = np.zeros((Nx,Ny))
    ax.Tsx = np.zeros((Nx,Ny))

    ax.Tnx[1:Nx,0:Ny-1] = Rho*(mixing_length[1:Nx,0:Ny-1]**2)*(dxx[1:Nx,0:Ny-1]/(dyy[1:Nx,1:Ny]**2))*abs(u[1:Nx,1:Ny]-u[1:Nx,0:Ny-1])
    ax.Tsx[0:Nx-1,1:Ny] = Rho*(mixing_length[0:Nx-1,1:Ny]**2)*(dxx[1:Nx,1:Ny]/(dyy[1:Nx,1:Ny]**2))*abs(u[1:Nx,0:Ny-1]-u[1:Nx,1:Ny])

    ax.aN += ax.Tnx
    ax.aS += ax.Tsx
    ax.aP += ax.Tnx + ax.Tsx

    # For v-momentum equation
    ay.Tey = np.zeros((Nx,Ny))
    ay.Twy = np.zeros((Nx,Ny))

    ay.Tey[0:Nx-1,0:Ny] = Rho*(mixing_length[0:Nx-1,0:Ny]**2)*(dyy[0:Nx-1,0:Ny]/(dxx[0:Nx-1,0:Ny]**2))*abs(v[1:Nx,0:Ny]-v[0:Nx-1,0:Ny]);
    ay.Twy[1:Nx,0:Ny]   = Rho*(mixing_length[1:Nx,0:Ny]**2)*(dyy[1:Nx,0:Ny]/(dxx[1:Nx,0:Ny]**2))*abs(v[0:Nx-1,0:Ny]-v[1:Nx,0:Ny])

    ay.aE += ay.Tey
    ay.aW += ay.Twy
    ay.aP += ay.Tey + ay.Twy


    if Linearversion :       
        # For u-momentum equation
        dax.S[0:Nx,1:Ny]   += ax.Tsx[0:Nx,1:Ny] 
        dax.N[0:Nx,0:Ny-1] += ax.Tnx[0:Nx,0:Ny-1]
        dax.P[0:Nx,0:Ny-1] += ax.Tnx[0:Nx,0:Ny-1] + ax.Tsx[0:Nx,0:Ny-1] 
   
        # For v-momentum equation
        day.E[0:Nx,1:Ny]   += ay.Tey[0:Nx,1:Ny] 
        day.W[0:Nx,0:Ny-1] += ay.Twy[0:Nx,0:Ny-1]
        day.P[0:Nx,0:Ny-1] += ay.Tey[0:Nx,0:Ny-1] + ay.Twy[0:Nx,0:Ny-1] 

    return ax, ay




def SpatialDiscr_Hybrid(Wp, sol, Linearversion):

    ax = type("ax", (object,), {})
    ay = type("ay", (object,), {})
    dax = type("dax", (object,), {})
    day = type("day", (object,), {})

    Nx     = Wp.mesh.Nx
    Ny     = Wp.mesh.Ny
    dxx    = Wp.mesh.dxx
    dyy    = Wp.mesh.dyy
    dxx2   = Wp.mesh.dxx2
    dyy2   = Wp.mesh.dyy2

    Rho    = Wp.site.Rho

    u      = sol.u
    v      = sol.v

    # Init
    ax.aE, ax.aW, ax.aS, ax.aN, ax.aP = [np.zeros((Nx, Ny)) for _ in range(5)]
    Fex, Fwx, Fsx, Fnx = [np.zeros((Nx, Ny)) for _ in range(4)]
    dFex, dFwx, dFnx, dFsx = [np.zeros((Nx, Ny)) for _ in range(4)]

    ay.aE, ay.aW, ay.aS, ay.aN, ay.aP = [np.zeros((Nx, Ny)) for _ in range(5)]
    Fey, Fwy, Fsy, Fny = [np.zeros((Nx, Ny)) for _ in range(4)]
    dFey, dFwy, dFny, dFsy = [np.zeros((Nx, Ny)) for _ in range(4)]


    ##  Setting the coefficients according to the hybrid differencing scheme ##

    ## x-direction
    # Define convection coefficients and its derivatives
    # Fex = c ( u_{i,J} + u_{i+1,J} )
    dFex[0:Nx-1, 0:Ny] = Rho * 0.5 * dyy2[0:Nx-1, 0:Ny]
    Fex[0:Nx-1,0:Ny] = dFex[0:Nx-1,0:Ny] * (u[1:Nx, 0:Ny] + u[0:Nx-1, 0:Ny])

    # Few = c ( u_{i,J} + u_{i-1,J} )
    dFwx[1:Nx, 0:Ny] = Rho * 0.5 * dyy2[1:Nx, 0:Ny]
    Fwx[1:Nx,0:Ny] = dFwx[1:Nx,0:Ny] * (u[1:Nx, 0:Ny] + u[0:Nx-1, 0:Ny])

    # Fnx = c ( v_{I-1,j+1} + v_{I,j+1} )
    dFnx[1:Nx, 0:Ny-1] = Rho * 0.5 * dxx[1:Nx, 0:Ny-1]
    Fnx[1:Nx,0:Ny-1] = dFnx[1:Nx,0:Ny-1] * (v[1:Nx, 1:Ny] + v[0:Nx-1, 1:Ny])

    # Fsx = c ( v_{I-1,j} + v_{I,j} )
    dFsx[1:Nx, 0:Ny] = Rho * 0.5 * dxx[1:Nx, 0:Ny]
    Fsx[1:Nx,0:Ny]  = dFsx[1:Nx,0:Ny] * (v[1:Nx, 0:Ny] + v[0:Nx-1, 0:Ny])

    ax.aE = np.maximum(np.maximum(-Fex, -0.5 * Fex), np.zeros((Nx, Ny)))
    ax.aW = np.maximum(np.maximum(Fwx, 0.5 * Fwx), np.zeros((Nx, Ny)))
    ax.aN = np.maximum(np.maximum(-Fnx, -0.5 * Fnx), np.zeros((Nx, Ny)))
    ax.aS = np.maximum(np.maximum(Fsx, 0.5 * Fsx), np.zeros((Nx, Ny)))
    ax.aP = ax.aW + ax.aE + ax.aS + ax.aN + Fex - Fwx + Fnx - Fsx

    if Linearversion :
        dax.E,dax.W,dax.N,dax.S,dax.P = [np.zeros((Nx, Ny)) for _ in range(5)]
        dax.SW,dax.NW,dax.SE,dax.NE   = [np.zeros((Nx, Ny)) for _ in range(4)]
        day.E,day.W,day.N,day.S,day.P = [np.zeros((Nx, Ny)) for _ in range(5)]
        day.SW,day.NW,day.SE,day.NE   = [np.zeros((Nx, Ny)) for _ in range(4)]

        # daxe/du_(i,J) = daxe/du_(i+1,J)
        dax.aE = ((-Fex >= (-Fex / 2)) * (-Fex > np.zeros((Nx, Ny))) * -dFex + 
                             ((-Fex / 2) > -Fex) * ((-Fex / 2) > np.zeros((Nx, Ny))) * -dFex / 2)
        # daxw/du_(i,J) = daxe/du_(i-1,J)
        dax.aW = ((Fwx >= (Fwx / 2)) * (Fwx > np.zeros((Nx, Ny))) * dFwx + 
                             ((Fwx / 2) > Fwx) * ((Fwx / 2) > np.zeros((Nx, Ny))) * dFwx / 2)
        # daxn/dv_(I,j+1) = daxn/dv_(I-1,j+1)
        dax.aN = ((-Fnx >= (-Fnx / 2)) * (-Fnx > np.zeros((Nx, Ny))) * -dFnx + 
                             ((-Fnx / 2) > -Fnx) * ((-Fnx / 2) > np.zeros((Nx, Ny))) * -dFnx / 2)
        # daxs/dv_(I,j) = daxs/dv_(I-1,j)
        dax.aS = ((Fsx >= (Fsx / 2)) * (Fsx > np.zeros((Nx, Ny))) * dFsx + 
                             ((Fsx / 2) > Fsx) * ((Fsx / 2) > np.zeros((Nx, Ny))) * dFsx / 2)
        
        # daPx/du_{i+1,J}
        dax.aPE = dax.aE + dFex
        # daPx/du_{i-1,J}
        dax.aPW = dax.aW - dFwx
        # daPx/dv_{I,j+1}
        dax.aPN = dax.aN + dFnx
        # daPx/dv_{I,j}             
        dax.aPS = dax.aS - dFsx
        # daPx/du_{i,J}
        dax.aPP = dax.aW + dax.aE - dFwx + dFex
        
        # Define derivatives for linearized model with ax = -aPx u_{i,J} + aEx u_{i+1,J} + aWx u_{i-1,J} + aNx u_{i,J+1} + aSx u_{i,J-1}
        
        # dax/du_(i-1,J) = aWx + daWx/du_{i-1,J} u_{i-1,J} - daPx/du_{i-1,J} u_{i,J}
        dax.W[1:Nx, 0:Ny] = ax.aW[1:Nx, 0:Ny] + dax.aW[1:Nx, 0:Ny] * u[0:Nx-1, 0:Ny] - dax.aPW[1:Nx, 0:Ny] * u[1:Nx, 0:Ny]

        # dax/du_(i,J-1) = aSx
        dax.S  = ax.aS
        
        # dax/du_(i,J)   = -aPx + daWx/du_{i,J} u_{i-1,J} + daEx/du_{i,J} u_{i+1,J} - daPx/du_{i,J} u_{i,J}
        dax.P[1:Nx-1, 0:Ny] = ax.aP[1:Nx-1, 0:Ny] - dax.aW[1:Nx-1, 0:Ny] * u[0:Nx-2, 0:Ny] \
            - dax.aE[1:Nx-1, 0:Ny] * u[2:Nx, 0:Ny] + dax.aPP[1:Nx-1, 0:Ny] * u[1:Nx-1, 0:Ny]

        # dax/du_(i,J+1)   = aNx
        dax.N = ax.aN
        
        # dax/du_(i+1,J) = daEx/du_{i+1,J} u_{i+1,J} + aEx - daPx/du_{i+1,J} u_{i,J}
        dax.E[0:Nx-1, 0:Ny] = ax.aE[0:Nx-1, 0:Ny] + dax.aE[0:Nx-1, 0:Ny] * u[1:Nx, 0:Ny] \
            - dax.aPE[0:Nx-1, 0:Ny] * u[0:Nx-1, 0:Ny]

        # dax/dv_(I-1,j) = daSx/dv_{I-1,J} u_{i,J-1} - daPx/dv_{I-1,J} u_{i,J}
        dax.SW[1:Nx, 1:Ny] = dax.aS[1:Nx, 1:Ny] * u[1:Nx, 0:Ny-1] - dax.aPS[1:Nx, 1:Ny] * u[1:Nx, 1:Ny]

        # dax/dv_(I-1,j+1) = daNx/dv_{I-1,j+1} u_{i,J+1} - daPx/dv_{I-1,j+1} u_{i,J}
        dax.NW[0:Nx-1, 0:Ny-1] = dax.aN[0:Nx-1, 0:Ny-1] * u[0:Nx-1, 1:Ny] - dax.aPN[0:Nx-1, 0:Ny-1] * u[0:Nx-1, 0:Ny-1]

        # dax/dv_(I,j)   = daSx/dv_{I,j} u_{I,j-1} - daPx/dv_{I,j} u_{i,J}
        dax.SE[0:Nx, 1:Ny] = dax.aS[0:Nx, 1:Ny] * u[0:Nx, 0:Ny-1] - dax.aPS[0:Nx, 1:Ny] * u[0:Nx, 1:Ny]

        # dax/dv_(I,j+1) = daNx/dv_{I,j+1} u_{i,J+1} - daPx/dv_{I,j+1} u_{i,J}
        dax.NE[0:Nx, 0:Ny-1] = dax.aN[0:Nx, 0:Ny-1] * u[0:Nx, 1:Ny] - dax.aPN[0:Nx, 0:Ny-1] * u[0:Nx, 0:Ny-1]


    ### y-direction
    # Define convection coefficients and its derivatives
    # Fey = c ( u_{i+1,J} + u_{i+1,J-1} )
    dFey[0:Nx-1, 1:Ny] = Rho * 0.5 * dyy[0:Nx-1, 1:Ny]
    Fey[0:Nx-1, 1:Ny] = dFey[0:Nx-1, 1:Ny] * (u[1:Nx, 1:Ny] + u[1:Nx, 0:Ny-1])

    # Fwy = c ( u_{i,J} + u_{i,J-1} )
    dFwy[0:Nx, 1:Ny] = Rho * 0.5 * dyy[0:Nx, 1:Ny]
    Fwy[0:Nx, 1:Ny] = dFwy[0:Nx, 1:Ny] * (u[0:Nx, 1:Ny] + u[0:Nx, 0:Ny-1])

    # Fny = c ( v_{I,j+1} + v_{I,j} )
    dFny[0:Nx, 0:Ny-1] = Rho * 0.5 * dxx2[0:Nx, 0:Ny-1]
    Fny[0:Nx, 0:Ny-1] = dFny[0:Nx, 0:Ny-1] * (v[0:Nx, 0:Ny-1] + v[0:Nx, 1:Ny])

    # Fsy = c ( v_{I,j-1} + v_{I,j} )
    dFsy[0:Nx, 1:Ny] = Rho * 0.5 * dxx2[0:Nx, 1:Ny]
    Fsy[0:Nx, 1:Ny] = dFsy[0:Nx, 1:Ny] * (v[0:Nx, 0:Ny-1] + v[0:Nx, 1:Ny])

    ay.aE = np.maximum(np.maximum(-Fey, -0.5 * Fey), np.zeros((Nx, Ny)))
    ay.aW = np.maximum(np.maximum(Fwy, 0.5 * Fwy), np.zeros((Nx, Ny)))
    ay.aN = np.maximum(np.maximum(-Fny, -0.5 * Fny), np.zeros((Nx, Ny)))
    ay.aS = np.maximum(np.maximum(Fsy, 0.5 * Fsy), np.zeros((Nx, Ny)))
    ay.aP = ay.aW + ay.aE + ay.aS + ay.aN + Fey - Fwy + Fny - Fsy


    if Linearversion:
        day.aE = (-Fey >= (-Fey/2)) * (-Fey > np.zeros((Nx,Ny))) * -dFey \
               + ( (-Fey/2) > -Fey ) * ( (-Fey/2) > np.zeros((Nx,Ny))) * -dFey / 2
        day.aW = (Fwy>=(Fwy/2)) * (Fwy>np.zeros((Nx,Ny))) * dFwy + \
                 ((Fwy/2)>Fwy) * ((Fwy/2)>np.zeros((Nx,Ny))) * dFwy / 2
        day.aN  = (-Fny>=(-Fny/2)) * (-Fny>np.zeros((Nx,Ny))) * -dFny + \
                 ((-Fny/2)>-Fny) * ((-Fny/2)>np.zeros((Nx,Ny))) * -dFny/2
        day.aS  = (Fsy>=(Fsy/2)) * (Fsy>np.zeros((Nx,Ny))) * dFsy + \
                  ((Fsy/2)>Fsy) * ((Fsy/2)>np.zeros((Nx,Ny))) * dFsy/2
        
        # daPy/du_(i+1,J-1)
        day.aPE = day.aE + dFey
        # daPy/du_{i,J-1}
        day.aPW = day.aW - dFwy
        # daPy/dv_{I,j-1}
        day.aPS = day.aS - dFsy
        # daPy/dv_{I,j+1}
        day.aPN = day.aN + dFny
        # daPy/dv_{I,j}
        day.aPP = day.aN + dFny + day.aS - dFsy
        
        # Define derivatives for linearized model with ay = -aPy v_{I,j} + aEy v_{I+1,J} + aWy v_{I-1,J} + aNy v_{I,j+1} + aSy v_{I,j-1}
        # day/du_(i,J-1) = daWy/du_{i,J-1} v_{I-1,j} - daPy/du_{i,J-1} v_{I,j}
        day.SW[1:Nx,0:Ny]  = day.aW[1:Nx,0:Ny]*v[0:Nx-1,0:Ny] - day.aPW[1:Nx,0:Ny]*v[1:Nx,0:Ny]
        
        # day/du_(i,J) = daWy/du_{i,J} v_{I-1,j} - daPy/du_{i,J} v_{I,j}
        day.NW[1:Nx,0:Ny]  = day.aW[1:Nx,0:Ny]*v[0:Nx-1,0:Ny] - day.aPW[1:Nx,0:Ny]*v[1:Nx,0:Ny]

        # day/du_(i+1,J-1) = daEy/du_{i+1,J-1} v_{I+1,j} - daPy/du_{i+1,J-1} v_{I,j}
        day.SE[0:Nx-1,0:Ny] = day.aE[0:Nx-1,0:Ny]*v[1:Nx,0:Ny]  - day.aPE[0:Nx-1,0:Ny]*v[0:Nx-1,0:Ny]

        # day/du_(i+1,J) = daEy/du_{i+1,J} v_{I+1,j} - daPy/du_{i+1,J} v_{I,j}
        day.NE[0:Nx-1,0:Ny] = day.aE[0:Nx-1,0:Ny]*v[1:Nx,0:Ny]  - day.aPE[0:Nx-1,0:Ny]*v[0:Nx-1,0:Ny]

        # day/dv_(I-1,j) = aWy
        day.W = ay.aW
        
        # day/dv_(I,j-1) = aSy + daSy/dv_{I,j-1} v_{I,j-1} - daPy/dv_{I,j-1} v_{I,j}
        day.S[0:Nx,1:Ny] = ay.aS[0:Nx,1:Ny] + day.aS[0:Nx,1:Ny]*v[0:Nx,0:Ny-1] - day.aPS[0:Nx,1:Ny]*v[0:Nx,1:Ny]

        # day/dv_(I,j) = daNy/dv_{I,j} v_{I,j+1} + daSy/dv_{I,j} v_{I,j-1} - daPy/dv_{I,j} v_{I,j} - aPy
        day.P[0:Nx,1:Ny-1] = -day.aN[0:Nx,1:Ny-1]*v[0:Nx,2:Ny] - day.aS[0:Nx,1:Ny-1]*v[0:Nx,0:Ny-2] + \
                              day.aPP[0:Nx,1:Ny-1]*v[0:Nx,1:Ny-1] + ay.aP[0:Nx,1:Ny-1]
        
        # day/dv_(I,j+1) = daNy/dv_{I,j+1} v_{I,j+1} + aNy - daPy/dv_{I,j+1} v_{I,j}
        day.N[0:Nx,0:Ny-1] = ay.aN[0:Nx,0:Ny-1] + day.aN[0:Nx,0:Ny-1]*v[0:Nx,1:Ny] - day.aPN[0:Nx,0:Ny-1]*v[0:Nx,0:Ny-1]

        # day/dv_(I+1,j) = aEy
        day.E  = ay.aE

    ## Turbulence model
    ax, ay = Turbulence(Wp, sol, ax, ay, dax, day, Linearversion)    

    output = type("output", (), {})
    output.ax = ax
    output.ay = ay

    if Linearversion:
        output.dax = dax
        output.day = day

    return output