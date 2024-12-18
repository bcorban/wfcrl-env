# This file contains
#   Compute_B1_B2_bc 
#   MakingSparseMatrix
#   Make_Ax_b
#   Computesol  
#   MapSolution 

#   MakingSparseMatrixl  (used in linearversion)
#   MakingSparseMatrixlo (used in linearversion)
#   Solution_space   (used if options.Projection)

import numpy as np
from scipy.sparse import block_diag, diags, csr_matrix, csgraph, spdiags, bmat
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import spsolve
#from scipy.linalg import solve
import time
import copy
import pypardiso

from wfcrl.simulators.wfsimpy.scr.SpatialDiscretization import SpatialDiscr_Hybrid
from wfcrl.simulators.wfsimpy.scr.SystemDescription import Dynamical, Actuator, BoundaryConditions, Updateboundaries
from wfcrl.simulators.wfsimpy.scr.sparse_null import spspaces


# -- make matrix B1, B2, bc : used in InitWFSim
def Compute_B1_B2_bc(Wp):
    Nx     = Wp.mesh.Nx
    Ny     = Wp.mesh.Ny
    dxx2   = Wp.mesh.dxx2
    dyy2   = Wp.mesh.dyy2
    Rho    = 1 # Wp.site.Rho
    u_Inf  = Wp.site.u_Inf

    Bm1 = Rho *  (diags(-np.ones((Nx-2)*(Ny-2)) * (dyy2[1:-1, 1:-1].T).flatten(order='F'), 
                        offsets=0, shape=((Nx-3)*(Ny-2),(Nx-2)*(Ny-2)))
               + diags( np.ones((Nx-2)*(Ny-2)) * (dyy2[1:-1, 1:-1].T).flatten(order='F'), 
                       offsets=Ny-2, shape=((Nx-3)*(Ny-2),(Nx-2)*(Ny-2))))
    #print(Bm1.shape)
    Bm2 = Rho *  (diags(-np.ones((Nx-2)*(Ny-2)) * (dxx2[1:-1, 1:-1].T).flatten(order='F'), 
                        offsets=0, shape=((Nx-2)*(Ny-2),(Nx-2)*(Ny-2)))
               + diags( np.ones((Nx-2)*(Ny-2)) * (dxx2[1:-1, 1:-1].T).flatten(order='F'), 
                       offsets=1, shape=((Nx-2)*(Ny-2),(Nx-2)*(Ny-2))))

    #Bm2 = np.delete(Bm2, np.arange(Ny-2-1, Bm2.shape[0], Ny-2), axis=0)
    #print(Bm2.shape)
    Bm2 = delete_rows_csr(Bm2.tocsr(), list(np.arange(Ny-2-1, Bm2.shape[0], Ny-2)))
    #print(Bm2.shape)
    
    B1 = Bm1.T
    B2 = Bm2.T
    #print(B1.shape, B2.shape)

    bc = np.zeros((Ny-2)*(Nx-2))
    bc[:Ny-2] = -Rho * u_Inf * dyy2[0, 1:-1]

    #B1[(Ny-2)*(Nx-3):(Ny-2)*(Nx-2), :] = 0  # u_{Nx,J}=u_{Nx-1,J}
    #print(B1.shape)
    
    #B1 = delete_rows_csr(B1.tocsr(), list(np.arange((Ny-2)*(Nx-3), (Ny-2)*(Nx-2)))) # not work, create shape problem

    rows_to_zero = list(np.arange((Ny-2)*(Nx-3), (Ny-2)*(Nx-2)))    
    B1 = B1.tolil()
    B1[rows_to_zero, :] = 0
    B1 = B1.tocsr()
    #print(B1.shape)

    rows_to_zero = [ kk*(Ny-2)-1 for kk in range(1, Nx-1)] + [kk*(Ny-2) for kk in range(Nx-2)]
    B2 = B2.tolil()
    B2[rows_to_zero, :] = 0
    B2 = B2.tocsr()

    #B2 = delete_rows_csr(B2.tocsr(), rows_to_zero) # not work, create shape problem
    #for kk in range(1, Nx-1):
    #    B2[kk*(Ny-2)-1, :] = 0  # v_{I,Ny}=v_{I,Ny-1}

    #for kk in range(Nx-2):
    #    B2[kk*(Ny-2), :] = 0  # v_{I,3}=v_{I,2} for I=2,3,...,Nx

    #B1 = csr_matrix(B1.T)
    #B2 = csr_matrix(B2.T)
    B1 = B1.T
    B2 = B2.T

    # print(B1.shape, B2.shape)
    return B1, B2, bc





# -- make sparse matrix
def MakingSparseMatrix(Nx, Ny, ax, ix, iy, q):
    # ix is the index where i begins
    # iy is the index where j begins
    
    ix -= 1  # adapted from matlab to python
    iy -= 1
    q -= 1  # attention to q: add -1 to ax.aP etc

    # Add central components
    nn = (Nx - ix - 1 + q) * (Ny - iy - 1 + q)
    Ax = spdiags(ax.aP[ix:(-q-1), iy:(-q-1)].T.flatten(order='F'), 
                 0, nn, nn)
    #Ax =  spdiags(vec(ax.aP(ix:end-q,iy:end-q)'),0,(Nx-ix-1+q)*(Ny-iy-1+q),(Nx-ix-1+q)*(Ny-iy-1+q));
    
    # Add north components
    # ann     =   vec([zeros(Nx-ix-1+q,1)   ax.aN(ix:end-q,iy:end-q-1) ]') ;
    ann = np.hstack([np.zeros((Nx - ix - 1 + q, 1)), 
                     ax.aN[ix:(-q-1), iy:(-1-q-1)]]).T.flatten(order='F') 
    Ax += -spdiags(ann, 1, nn, nn)
    # print(Ax.toarray()[0,:])

    # Add south components
    # ass     =   vec([ax.aS(ix:end-q,iy+1:end-q) zeros(Nx-ix-1+q,1) ]') ;
    ass = np.hstack([ax.aS[ix:(-q-1), iy + 1:(-q-1)], 
                     np.zeros((Nx - ix - 1 + q , 1))]).T.flatten(order='F') 
    Ax += -spdiags(ass, -1, nn,nn)

    # Add east components
    # aee     =   vec([ zeros(1,Ny-iy-1+q);ax.aE(ix:end-q-1,iy:end-q);]');
    aee = np.vstack([np.zeros((1, Ny - iy - 1 + q)), 
                     ax.aE[ix:(-1-q-1), iy:(-q-1)]]).T.flatten(order='F')  
    Ax += -spdiags(aee, Ny - iy - 1 + q, nn, nn)

    # Add west components
    # aww     =   vec([ax.aW(ix+1:end-q,iy:end-q);zeros(1,Ny-iy-1+q)]');
    aww = np.vstack((ax.aW[ix + 1:(-q-1), iy:(-q-1)], 
                     np.zeros((1, Ny - iy - 1 + q )))).T.flatten(order='F')  
    Ax += -spdiags(aww, -(Ny - iy - 1 + q), nn, nn)

    return Ax





# -- Create system matrices sys.A and sys.b for our nonlinear model,
#    where WFSim basically comes down to: sys.A*sol.x=sys.b.
def Make_Ax_b(Wp, sys, sol, options):
    # Import variables
    Nx    = Wp.mesh.Nx
    Ny    = Wp.mesh.Ny

    # Decide whether to start from uniform flow field or steady state
    if sol.k == 1 and Wp.sim.startUniform == 0 :
        dt = np.inf
    else :
        dt = Wp.sim.h
        dt = dt/2     # correction factor for difference with LES wake propagation

    # --  creating the system matrices
    # Spatial discretization
    StrucDiscretization = SpatialDiscr_Hybrid(Wp, sol, options.Linearversion) 

    # Dynamical term
    StrucDiscretization, StrucDynamical = Dynamical(Wp, StrucDiscretization, sol, dt, options.Linearversion)
    # Actuator/forcing function
    StrucActuator, sol   = Actuator(Wp, sol, options)
    # Zero gradient boundary conditions momentum equations
    StrucDiscretization, StrucBCs = BoundaryConditions(Wp, StrucDiscretization, sol, options.Linearversion) 
    #print(StrucDiscretization.ax.aS[:,0])
    #print(StrucDiscretization.ax.aN[:,0])

    # Collect all terms and create the A matrix in 'A*x = b' 
    Ax    = MakingSparseMatrix(Nx, Ny, StrucDiscretization.ax, 3, 2, 1)
    Ay    = MakingSparseMatrix(Nx, Ny, StrucDiscretization.ay, 2, 3, 1)
    
    # Création de la matrice A
    A1 = block_diag([Ax, Ay])
    # print(Ax.toarray()[0,0:5])
    B1_B2 = bmat([[sys.B1], [sys.B2]])
    B1_B2_transpose = bmat([[sys.B1], [2*sys.B2]]).T
    sparse_matrix = csr_matrix(((Nx-2)*(Ny-2), (Nx-2)*(Ny-2)), dtype=np.float32)
    sys.A = bmat([[A1, B1_B2], [B1_B2_transpose, sparse_matrix]])

    '''
    # saved to debug
    sys.Ax = Ax
    sys.Ay = Ay
    '''


    # If necessary, project away the continuity equation in A*x = b            
    if  options.Projection :
        # Création de sys.Ct
        sys.Ct = block_diag(
            diags(StrucDynamical.ccx, 0, shape=(Wp.Nu, Wp.Nu)),
            diags(StrucDynamical.ccy, 0, shape=(Wp.Nv, Wp.Nv))
        )
        # Calcul de sys.Et
        sys.Et = sys.Qsp.T @ (block_diag(Ax, Ay) @ sys.Qsp)
        # Calcul de sys.At
        sys.At = sys.Qsp.T @ (sys.Ct @ sys.Qsp)
        # Calcul de sys.St
        sys.St = (sys.Qsp.T @ np.hstack([StrucBCs.bx, StrucBCs.by]) -
                sys.Qsp.T @ block_diag(Ax, Ay) @ sys.Bsp +
                sys.Qsp.T @ (sys.Ct @ sys.Bsp))
        # Calcul de sys.Bt
        sys.Bt = sys.Qsp.T @ np.hstack([StrucActuator.Sm.xx, StrucActuator.Sm.yy])


        if sol.k==1 :
            # Calculate RCM

            # Obtenir l'ordre de permutation
            perm = reverse_cuthill_mckee(sys.Et, symmetric_mode=False)
            # Si vous souhaitez réordonner la matrice
            sys.pRCM = sys.Et[perm][:, perm]

        
        if options.Linearversion :
            Ayl       = MakingSparseMatrixl(Nx,Ny,StrucDiscretization.day,2,3,1)
            Axl       = MakingSparseMatrixl(Nx,Ny,StrucDiscretization.dax,3,2,1)
            Axlo,Aylo = MakingSparseMatrixlo(Nx,Ny,StrucDiscretization.dax,StrucDiscretization.day)

            sys.Al      = np.block([[np.hstack([Axl,Axlo,sys.B1]), np.hstack([Aylo,Ayl,sys.B2])],
                                   [np.hstack([sys.B1.T, 2@sys.B2.T]), sparse_matrix]]) - sys.A
            
            sys.Atl = StrucDynamical.dcdx + StrucBCs.dbcdx + StrucActuator.dSm.dx - sys.Al
            sys.Atl = sys.Qsp.T@sys.Atl[0:(Nx-3)*(Ny-2)+(Nx-2)*(Ny-3),0:(Nx-3)*(Ny-2)+(Nx-2)*(Ny-3)]@sys.Qsp
            sys.Etl = sys.Et
            sys.Btl = sys.Qsp.T@ np.hstack([StrucActuator.Sm.dxx, StrucActuator.Sm.dyy])
        
    else:
        # Collect all terms and create the b vector in 'A*x = b'
        sys.b    = np.hstack([StrucBCs.bx + StrucDynamical.cx + StrucActuator.Sm.x.T.toarray().flatten(order='F'),
                              StrucBCs.by + StrucDynamical.cy + StrucActuator.Sm.y.T.toarray().flatten(order='F'),
                              sys.bc])
        
        '''
        # saved to debug
        sol.StrucBCs = StrucBCs
        sol.StrucDynamical = StrucDynamical
        sol.StrucActuator = StrucActuator
        sol.StrucDiscretization = StrucDiscretization
        '''

        if options.Linearversion:
            # Linear version
            Ayl       = MakingSparseMatrixl(Nx,Ny,StrucDiscretization.day,2,3,1)
            Axl       = MakingSparseMatrixl(Nx,Ny,StrucDiscretization.dax,3,2,1)
            Axlo,Aylo = MakingSparseMatrixlo(Nx,Ny,StrucDiscretization.dax,StrucDiscretization.day)

            Al = np.block([[Axl, Axlo, sys.B1],
                           [Aylo, Ayl, sys.B2],
                           [sys.B1.T, 2 * sys.B2.T, 
                            csr_matrix(((Nx - 2) * (Ny - 2), (Nx - 2) * (Ny - 2)), dtype=np.float32)]]) -sys.A
            
            sys.Al = (StrucDynamical.dcdx + StrucBCs.dbcdx + StrucActuator.dSm.dx - Al)
            sys.Bl = np.vstack((StrucActuator.Sm.dxx, StrucActuator.Sm.dyy, np.zeros((len(sys.bc), Wp.turbine.N * 2))))
            sys.bl = np.concatenate((StrucActuator.Sm.dx.flatten(order='F'),
                                     StrucActuator.Sm.dy.flatten(order='F'),
                                     np.zeros(len(sys.bc)) ))
            
            # Indice pour la suppression des lignes et colonnes
            index_to_remove = Ax.get_shape()[0] + Ay.get_shape()[0] + sys.B1.get_shape()[0]  - (Ny-2)
            #size(Ax, 0) + size(Ay, 0) + size(sys.B1, 0) - (Ny - 2)

            # Suppression des lignes et colonnes dans sys.Al, sys.bl, sys.Bl
            sys.Al = np.delete(sys.Al, index_to_remove, axis=0)
            sys.bl = np.delete(sys.bl, index_to_remove, axis=0)
            sys.Bl = np.delete(sys.Bl, index_to_remove, axis=0)
            # Suppression de la colonne
            sys.Al = np.delete(sys.Al, index_to_remove, axis=1)
            sys.Al = np.delete(sys.Al, -1, axis=1)  # Dernière colonne
            sys.Al = np.delete(sys.Al, -1, axis=0)  # Dernière ligne
            sys.bl = np.delete(sys.bl, -1, axis=0)
            sys.Bl = np.delete(sys.Bl, -1, axis=0)  # Dernière ligne

        
        # Répétition pour sys.A et sys.b
        index_to_remove = Ax.shape[0] + Ay.shape[0] + sys.B1.T.shape[0]-(Ny-2)
        sys.A = delete_rows_csr(sys.A.tocsr(), [index_to_remove])
        sys.A = delete_column_csr(sys.A.tocsr(), [index_to_remove])
        sys.A = delete_column_csr(sys.A.tocsr(), [-1])   # Dernière colonne
        sys.A = delete_rows_csr(sys.A.tocsr(), [-1])     # Dernière ligne

        sys.b = delete_rows_csr(csr_matrix(sys.b).T, [index_to_remove])
        sys.b = delete_rows_csr(sys.b.tocsr(), [-1])

        #print(f'shape of sys.A = {sys.A.shape}')
        #print(f'shape of sys.b = {sys.b.shape}')

        if sol.k==1:
            sys.pRCM = reverse_cuthill_mckee(sys.A, symmetric_mode=False)

    return sol, sys



def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]

def delete_column_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    indices = list(indices)
    mask = np.ones(mat.shape[1], dtype=bool)
    mask[indices] = False
    return mat[:,mask]




# --  This function solves A*x=b for x to obtain the flow fields.
def Computesol(sys, sol, it, options):

    # Import variables
    k               = sol.k
    input           = sol.turbInput
    Projection      = options.Projection
    exportLinearSol = options.exportLinearSol

    # Find the solution
    if Projection:
        # Résoudre en projetant l'équation de continuité
        if it == 1 and k == 1:
            # Définir la condition initiale
            sol.alpha = spsolve(sys.Qsp, np.hstack((sol.u[2:-1, 1:-1].ravel(), sol.v[1:-1, 2:-1].ravel())) - sys.Bsp)

        beta = 1/4 * input.CT_prime
        Ft = sys.At @ sol.alpha + sys.Bt @ np.hstack((beta, input.phi)) + sys.St
        sol.alpha[sys.pRCM, 0] = spsolve(sys.Et[sys.pRCM][:, sys.pRCM], Ft[sys.pRCM])

        sol.x = sys.Qsp @ sol.alpha + sys.Bsp

        if exportLinearSol:
            if it == 1 and k == 1:
                sol.dalpha = np.zeros((sys.Qsp.shape[1], 1))
            dbeta = 1/4 * input.dCT_prime
            Ftl = sys.Atl @ sol.dalpha + sys.Btl @ np.hstack((dbeta, input.dphi))
            sol.dalpha[sys.pRCM, 0] = spsolve(sys.Etl[sys.pRCM][:, sys.pRCM], Ftl[sys.pRCM])
            sol.dx = sys.Qsp @ sol.dalpha

    else:
        # Sinon, c'est la solution simple x = A\b
        #sol.x = np.zeros(len(sys.pRCM))
        #sol.x[sys.pRCM] = spsolve(sys.A[sys.pRCM][:, sys.pRCM], sys.b[sys.pRCM])
        #sol.x = spsolve(sys.A, sys.b)  #slower
        sol.x = np.zeros(len(sys.pRCM), dtype=np.float32)
        sol.x[sys.pRCM] = spsolve(sys.A[sys.pRCM][:, sys.pRCM], sys.b[sys.pRCM])
        
        # use cupy to accelerate ?

        # sol.x = solve( sys.A.toarray(order='F'),
        #                sys.b.toarray(order='F'), 
        #                overwrite_a=True, overwrite_b=True )
        # this is 100 times slower

        if exportLinearSol:
            if it == 1 and k == 1:
                sol.dx = np.zeros((sys.Al.shape[1], 1))
            bll = sys.Al @ sol.dx + sys.bl
            sol.dx[sys.pRCM, 0] = spsolve(sys.A[sys.pRCM][:, sys.pRCM], bll[sys.pRCM])

    return sol, sys


# -- This function converts 'sol.x' to real flow fields.
def MapSolution(Wp, sol, it, options):

    # Import variables
    k  = sol.k
    Nx = Wp.mesh.Nx
    Ny = Wp.mesh.Ny
    exportPressures = options.exportPressures
    exportLinearSol = options.exportLinearSol

    # Project sol.x back to the flow fields, excluding the boundary conditions
    sol.uu = np.zeros((Nx,Ny), dtype=np.float32)
    sol.vv = np.zeros((Nx,Ny), dtype=np.float32)
    sol.pp = np.zeros((Nx,Ny), dtype=np.float32)
    u_ = copy.deepcopy(sol.u)
    v_ = copy.deepcopy(sol.v)
    p_ = copy.deepcopy(sol.p)

    sol.uu[2:-1, 1:-1] = np.reshape(sol.x[0:(Nx-3)*(Ny-2)], (Ny-2, Nx-3), order='F').T
    sol.vv[1:-1, 2:-1] = np.reshape(sol.x[(Nx-3)*(Ny-2):(Nx-3)*(Ny-2)+(Nx-2)*(Ny-3)], 
                                    (Ny-3, Nx-2), order='F').T

    if exportPressures:
        sol.pp[1:-1, 1:-1] = np.reshape(np.concatenate((sol.x[(Nx-3)*(Ny-2)+(Nx-2)*(Ny-3):], 
                                                        [0, 0])), 
                                                        (Ny-2, Nx-2), order='F').T
        sol.pp[np.isinf(sol.pp)] = 0

    if exportLinearSol:
        sol.du[2:-1, 1:-1] = np.reshape(sol.dx[0:(Nx-3)*(Ny-2)], 
                                        (Ny-2, Nx-3), order='F').T
        sol.dv[1:-1, 2:-1] = np.reshape(sol.dx[(Nx-3)*(Ny-2):(Nx-3)*(Ny-2)+(Nx-2)*(Ny-3)], 
                                        (Ny-3, Nx-2), order='F').T
        sol.ul[2:-1, 1:-1] += sol.du[2:-1, 1:-1]
        sol.vl[1:-1, 2:-1] += sol.dv[1:-1, 2:-1]

        if exportPressures:
            sol.dp[1:-1, 1:-1] = np.reshape(np.concatenate((sol.dx[(Nx-3)*(Ny-2)+(Nx-2)*(Ny-3):], 
                                                            [0, 0])), (Ny-2, Nx-2), order='F').T
            sol.dp[np.isinf(sol.dp)] = 0

            sol.pl[1:-1, 1:-1] = np.reshape(np.concatenate((sol.dx[(Nx-3)*(Ny-2)+(Nx-2)*(Ny-3):],
                                                            [0, 0])), (Ny-2, Nx-2), order='F').T
            sol.pl[np.isinf(sol.pl)] = 0

    # Check if solution has converged
    Normv = np.linalg.norm(sol.v[1:-1, 2:-1] - sol.vv[1:-1, 2:-1])
    Normu = np.linalg.norm(sol.u[2:-1, 1:-1] - sol.uu[2:-1, 1:-1])
    eps = np.sqrt((Normv + Normu)) / ((Ny-2) * (Nx-2)) / 2

    alpha = min(1 - 0.9**it, 1) if k == 1 else 1

    u_ [2:-1, 1:-1] = (1 - alpha) * sol.u[2:-1, 1:-1] + alpha * sol.uu[2:-1, 1:-1]
    v_[1:-1, 2:-1] = (1 - alpha) * sol.v[1:-1, 2:-1] + alpha * sol.vv[1:-1, 2:-1]
    p_[1:-1, 1:-1] = (1 - alpha) * sol.p[1:-1, 1:-1] + alpha * sol.pp[1:-1, 1:-1]

    #sol.u[2:-1, 1:-1] = u_
    #sol.v[1:-1, 2:-1] = v_
    #sol.p[1:-1, 1:-1] = p_
    del sol.uu, sol.vv, sol.pp

    # Update velocities for next iteration and boundary conditions
    sol.u, sol.v, sol.p = Updateboundaries(Nx, Ny, u_, v_, p_)

    del u_, v_, p_

    if exportLinearSol:
        sol.ul, sol.vl, sol.pl = Updateboundaries(Nx, Ny, sol.ul, sol.vl, sol.pl)
        sol.du, sol.dv, sol.dp = Updateboundaries(Nx, Ny, sol.du, sol.dv, sol.dp)

    if options.printConvergence:
        print(f'k {sol.k:.0f}, It {it:.0f}, Nv={Normv:10.2e}, Nu={Normu:10.2e}, TN={eps:10.2e}')

    return sol,eps



## -- functions used in case of Linearversion -- ##
def MakingSparseMatrixl(Nx,Ny,dax,ix,iy,q):
    # ix is the index where i begins
    # iy is the index where j begins

    ix -= 1
    iy -= 1
    q -= 1

    # Add central components
    Ax_on = spdiags(dax.P[ix:(-q-1), iy:(-q-1)].flatten(order='F'), 
                    0, (Nx - ix - 1 + q) * (Ny - iy - 1 + q), (Nx - ix - 1 + q) * (Ny - iy - 1 + q))

    # Add north components
    ann = np.vstack((np.zeros((Nx - ix - 1 + q, 1)), dax.N[ix:(-q-1), iy:-(q+1-1)]))
    Ax_on += -spdiags(ann.flatten(order='F'), 
                      1, (Nx - ix - 1 + q) * (Ny - iy - 1 + q), (Nx - ix - 1 + q) * (Ny - iy - 1 + q))

    # Add south components
    ass = np.vstack((dax.S[ix:(-q-1), iy + 1:(-q-1)], np.zeros((Nx - ix - 1 + q, 1))))
    Ax_on += -spdiags(ass.flatten(order='F'), 
                      -1, (Nx - ix - 1 + q) * (Ny - iy - 1 + q), (Nx - ix - 1 + q) * (Ny - iy - 1 + q))

    # Add east components
    aee = np.hstack((np.zeros((1, Ny - iy - 1 + q)), dax.E[ix:-(q+1-1), iy:(-q-1)]))
    Ax_on += -spdiags(aee.flatten(order='F'), Ny - iy - 1 + q, (Nx - ix - 1 + q) * (Ny - iy - 1 + q), (Nx - ix - 1 + q) * (Ny - iy - 1 + q))

    # Add west components
    aww = np.hstack((dax.W[ix + 1:(-q-1), iy:(-q-1)], np.zeros((1, Ny - iy - 1 + q))))
    Ax_on += -spdiags(aww.flatten(order='F'), -(Ny - iy - 1 + q), (Nx - ix - 1 + q) * (Ny - iy - 1 + q), (Nx - ix - 1 + q) * (Ny - iy - 1 + q))
    
    return Ax_on


def MakingSparseMatrixlo(Nx,Ny,dax,day):

    Ax_off = csr_matrix(((Ny - 2) * (Nx - 3), (Nx - 2) * (Ny - 3)))
    Ay_off = csr_matrix(((Ny - 3) * (Nx - 2), (Ny - 2) * (Nx - 3)))


    for x in range(2, Nx-1):
        swx = -spdiags(dax.SW[x, 2:Ny-1], 0, Ny - 3, Ny - 3)
        nwx = -spdiags(dax.NW[x, 1:Ny-2], 0, Ny - 3, Ny - 3)

        Bx_W = csr_matrix(np.vstack((np.zeros((1, Ny - 3)), swx.toarray())) + 
                        np.vstack((nwx.toarray(), np.zeros((1, Ny - 3)))))

        sex = -spdiags(dax.SE[x, 2:Ny-1], 0, Ny - 3, Ny - 3)
        nex = -spdiags(dax.NE[x, 1:Ny-2], 0, Ny - 3, Ny - 3)
        Bx_E = csr_matrix(np.vstack((np.zeros((1, Ny - 3)), sex.toarray())) + 
                        np.vstack((nex.toarray(), np.zeros((1, Ny - 3)))))

        Ax_off[(x - 3) * (Ny - 2):(x - 2) * (Ny - 2), 
            (x - 3) * (Ny - 3):(x - 1) * (Ny - 3)] = csr_matrix(np.hstack((Bx_W.toarray(), Bx_E.toarray())))
 

    for y in range(2, Nx-1):
        swy = -spdiags(day.SW[y, 2:Ny-1], 0, Ny - 3, Ny - 3)
        nwy = -spdiags(day.NW[y, 2:Ny-1], 0, Ny - 3, Ny - 3)
        By_W = csr_matrix(np.hstack((swy.toarray(), np.zeros((Ny - 3, 1)))) + 
                        csr_matrix(np.hstack((np.zeros((Ny - 3, 1)), nwy.toarray()))))

        sey = -spdiags(day.SE[y - 1, 2:Ny-1], 0, Ny - 3, Ny - 3)
        ney = -spdiags(day.NE[y - 1, 2:Ny-1], 0, Ny - 3, Ny - 3)
        By_E = csr_matrix(np.hstack((sey.toarray(), np.zeros((Ny - 3, 1)))) + 
                        csr_matrix(np.hstack((np.zeros((Ny - 3, 1)), ney.toarray()))))

        Ay_off[(y - 3) * (Ny - 3):(y - 1) * (Ny - 3), 
            (y - 3) * (Ny - 2):(y - 2) * (Ny - 2)] = csr_matrix(np.vstack((By_E.toarray(), By_W.toarray())))

    return Ax_off,Ay_off



## -- used if options.Projection is true
def Solution_space(B1, B2, bc):

    A = np.vstack((B1.T, B2.T))

    #[~,SRight] = spspaces(A,2);
    SRight = spspaces(A, 2)  # Vous devez définir cette fonction

    Qsp1 = SRight[0][:, SRight[2]]
    Qsp = csr_matrix((Qsp1.shape[0], Qsp1.shape[1]))

    # Sparsification
    eps=1e-6

    for i in range(Qsp1.shape[1]):
        ind = np.where(np.sign(np.abs(Qsp1[:, i]) - eps) + 1)[0]
        Qsp[ind, i] = Qsp1[ind, i]

    Bsp = np.linalg.solve(A[:-1, :], bc[:-1, :])

    # Check
    print(np.count_nonzero(Qsp.T @ np.vstack((B1, B2))))  # équation de momentum
    print(np.count_nonzero((np.vstack((B1.T, B2.T)) @ Qsp)))  # équation de continuité

    del Qsp1, SRight

    ##
    SRight = spspaces(B1.T, 2)  # Vous devez définir cette fonction
    PP = SRight[0][:, SRight[2]]
    P1 = csr_matrix((PP.shape[0], PP.shape[1]))  # P1 est vide car B1' est de rang complet

    # Sparsification
    for i in range(PP.shape[1]):
        ind = np.where(np.sign(np.abs(PP[:, i]) - eps) + 1)[0]
        P1[ind, i] = PP[ind, i]

    del PP, SRight

    SRight = spspaces(B2.T, 2)  # Vous devez définir cette fonction
    PP = SRight[0][:, SRight[2]]
    P2 = csr_matrix((PP.shape[0], PP.shape[1]))  # B2' n'est pas de rang complet

    # Sparsification
    for i in range(PP.shape[1]):
        ind = np.where(np.sign(np.abs(PP[:, i]) - eps) + 1)[0]
        P2[ind, i] = PP[ind, i]

    del PP, SRight

    P1 = csr_matrix((P1.shape[0], P2.shape[1]))  # This since B1' does not have a nullspace
    P = np.vstack((P1, P2))

    # Check
    print(np.count_nonzero(P.T @ np.vstack((B1, B2))))  # équation de momentum
    print(np.count_nonzero((np.vstack((B1.T, B2.T)) @ P)))  # équation de continuité

    # Qsp = P;
    return Qsp, Bsp