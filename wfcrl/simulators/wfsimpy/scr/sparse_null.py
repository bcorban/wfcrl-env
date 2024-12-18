# This file contains functions
#  spspaces
#  luq

import numpy as np
from scipy.linalg import lu
from scipy.sparse import csr_matrix, eye

def spspaces(A, opt, tol=None):
    '''
    %  PURPOSE: finds left and right null and range space of a sparse matrix A
    %
    % ---------------------------------------------------
    %  USAGE: [SpLeft, SpRight] = spspaces(A,opt,tol)
    %
    %  INPUT: 
    %       A                           a sparse matrix
    %       opt                         spaces to calculate
    %                                   = 1: left null and range space
    %                                   = 2: right null and range space
    %                                   = 3: both left and right spaces
    %       tol                         uses the tolerance tol when calculating
    %                                   null subspaces (optional)
    %
    %   OUTPUT:
    %       SpLeft                      1x4 cell. SpLeft = {} if opt =2.
    %           SpLeft{1}               an invertible matrix Q
    %           SpLeft{2}               indices, I, of rows of the matrix Q that
    %                                   span the left range of the matrix A
    %           SpLeft{3}               indices, J, of rows of the matrix Q that
    %                                   span the left null space of the matrix A
    %                                   Q(J,:)A = 0
    %           SpLeft{4}               inverse of the matrix Q
    %       SpRight                     1x4 cell. SpRight = {} if opt =1.
    %           SpLeft{1}               an invertible matrix Q
    %           SpLeft{2}               indices, I, of rows of the matrix Q that
    %                                   span the right range of the matrix A
    %           SpLeft{3}               indices, J, of rows of the matrix Q that
    %                                   span the right null space of the matrix A
    %                                   AQ(:,J) = 0
    %           SpLeft{4}               inverse of the matrix Q
    %
    %   COMMENTS:
    %       uses luq routine, that finds matrices L, U, Q such that
    %
    %           A = L | U 0 | Q
    %                 | 0 0 |
    %       
    %       where L, Q, U are invertible matrices, U is upper triangular. This
    %       decomposition is calculated using lu decomposition.
    %
    %       This routine is fast, but can deliver inaccurate null and range
    %       spaces if zero and nonzero singular values of the matrix A are not
    %       well separated.
    %
    %   WARNING:
    %       right null and rang space may be very inaccurate
    %
    % Copyright  (c) Pawel Kowal (2006)
    % All rights reserved
    % LREM_SOLVE toolbox is available free for noncommercial academic use only.
    % pkowal3@sgh.waw.pl
    '''
    if tol is None:
        tol = max(max(A.shape) * np.linalg.norm(A, 1) * np.finfo(float).eps, 100 * np.finfo(float).eps)


    if opt == 1:
        calc_left = True
        calc_right = False
    elif opt == 2:
        calc_left = False
        calc_right = True
    elif opt == 3:
        calc_left = True
        calc_right = True

    # Décomposition LU
    P, L, U = luq(A,0,tol)

    SpLeft = []
    if calc_left:
        if L.size > 0:
            LL = np.linalg.inv(L)
        else:
            LL = L
        
        S = np.max(np.abs(U), axis=1)
        I = np.where(S > tol)[0]
        if S.size > 0:
            J = np.where(S <= tol)[0]
        else:
            J = np.arange(S.size)
        
        SpLeft = [LL, I, J, L]


    SpRight = []
    if calc_right:
        if Q.size > 0:  # Assuming Q is derived from the permutation matrix P
            QQ = np.linalg.inv(P)
        else:
            QQ = P
        
        S = np.max(np.abs(U), axis=0)
        I = np.where(S > tol)[0]
        if S.size > 0:
            J = np.where(S <= tol)[0]
        else:
            J = np.arange(S.size)
        
        SpRight = [QQ, I, J, P]

    return SpLeft, SpRight


def luq(A,do_pivot,tol):
    '''
    %  PURPOSE: calculates the following decomposition
    %             
    %       A = L |Ubar  0 | Q
    %             |0     0 |
    %
    %       where Ubar is a square invertible matrix
    %       and matrices L, Q are invertible.
    %
    % ---------------------------------------------------
    %  USAGE: [L,U,Q] = luq(A,do_pivot,tol)
    %  INPUT: 
    %         A             a sparse matrix
    %         do_pivot      = 1 with column pivoting
    %                       = 0 without column pivoting
    %         tol           uses the tolerance tol in separating zero and
    %                       nonzero values
    %
    %   OUTPUT:
    %         L,U,Q          matrices
    %
    %   COMMENTS:
    %         based on lu decomposition
    %
    % Copyright  (c) Pawel Kowal (2006)
    % All rights reserved
    % LREM_SOLVE toolbox is available free for noncommercial academic use only.
    % pkowal3@sgh.waw.pl
    '''

    n, m = A.shape

    if not isinstance(A, csr_matrix):
        A = csr_matrix(A)

    #--------------------------------------------------------------------------
    #       SPECIAL CASES
    #--------------------------------------------------------------------------
    if n == 0:
        L = eye(n, format='csr')
        U = A
        Q = eye(m, format='csr')
        return L, U, Q
    
    if m == 0:
        L = eye(n, format='csr')
        U = A
        Q = eye(m, format='csr')
        return L, U, Q  

    #--------------------------------------------------------------------------
    #       LU DECOMPOSITION
    #--------------------------------------------------------------------------
    if do_pivot:
        P, L, U = lu(A.toarray())
        Q = np.transpose(P)  # Adjusting for the permutation
    else:
        P, L, U = lu(A.toarray())
        Q = eye(m)
    
    p = n - L.shape[1]
    LL = csr_matrix((n - p, p)).tolil()                    # Sparse matrix with zero entries
    L = csr_matrix(np.hstack([P.T @ L, L[n - p:n, :].T]))  # Adjust L
    U = csr_matrix(np.vstack([U, csr_matrix((p, m))]))     # Adjust U

    #--------------------------------------------------------------------------
    #       FINDS ROWS WITH ZERO AND NONZERO ELEMENTS ON THE DIAGONAL
    #--------------------------------------------------------------------------
    if U.shape[0] == 1 or U.shape[1] == 1:
        S = U[0, 0]
    else:
        S = np.diag(U.toarray())

    I = np.where(np.abs(S) > tol)[0]
    Jl = np.arange(n)
    Jl = np.delete(Jl, I)
    Jq = np.arange(m)
    Jq = np.delete(Jq, I)

    Ubar1 = U[I[:, None], I]
    Ubar2 = U[Jl[:, None], Jq]
    Qbar1 = Q[I[:, None], :]
    Lbar1 = L[:, I]

    #--------------------------------------------------------------------------
    #       ELININATES NONZEZO ELEMENTS BELOW AND ON THE RIGHT OF THE
    #       INVERTIBLE BLOCK OF THE MATRIX U
    #
    #       UPDATES MATRICES L, Q
    #--------------------------------------------------------------------------
    if I.size > 0:
        Utmp = U[I, :][:, Jq]
        X = np.linalg.solve(Ubar1.toarray().T, U[Jl, :][:, I].T)
        Ubar2 -= X.T @ Utmp.toarray()
        Lbar1 += L[:, Jl] @ X.T

        X = np.linalg.solve(Ubar1.toarray(), Utmp.toarray())
        Qbar1 += X @ Q[Jq, :]
        Utmp = None
        X = None

    #--------------------------------------------------------------------------
    #       FINDS ROWS AND COLUMNS WITH ONLY ZERO ELEMENTS
    #--------------------------------------------------------------------------
    I2 = np.where(np.max(np.abs(Ubar2.toarray()), axis=1) > tol)[0]
    I5 = np.where(np.max(np.abs(Ubar2.toarray()), axis=0) > tol)[0]

    I3 = Jl[I2]
    I4 = Jq[I5]
    Jq = np.delete(Jq, I5)
    Jl = np.delete(Jl, I2)
    U = None

    #--------------------------------------------------------------------------
    #       FINDS A PART OF THE MATRIX U WHICH IS NOT IN THE REQIRED FORM
    #--------------------------------------------------------------------------
    A = Ubar2[I2[:, None], I5]

    #--------------------------------------------------------------------------
    #       PERFORMS LUQ DECOMPOSITION OF THE MATRIX A
    #--------------------------------------------------------------------------
    L1, U1, Q1 = luq(A, do_pivot, tol)

    #--------------------------------------------------------------------------
    #       UPDATES MATRICES L, U, Q
    #--------------------------------------------------------------------------
    Lbar2 = L[:, I3] @ L1
    Qbar2 = Q1 @ Q[I4, :]
    L = csr_matrix(np.hstack([Lbar1, Lbar2, L[:, Jl]]))
    Q = csr_matrix(np.vstack([Qbar1, Qbar2, Q[Jq, :]]))

    n1 = len(I)
    n2 = len(I3)
    m2 = len(I4)
    U = csr_matrix(np.vstack([
        csr_matrix(np.hstack([Ubar1, csr_matrix((n1, m - n1))])),
        csr_matrix(np.hstack([csr_matrix((n2, n1)), U1, csr_matrix((n2, m - n1 - m2))])),
        csr_matrix((n - n1 - n2, m))
    ]))
    return L,U,Q


if __name__ == '__main__':
    import numpy as np
    from scipy.sparse import diags, eye, random
    from scipy.sparse.linalg import inv
    import time

    # SET PARAMETERS
    n           = 10000                #matrix dimension
    k           = 30                   #null space dimension
    K           = .2                   #controls matrix sparsity
    smax        = 10^6                 #maximal singular value of the matrix
    smin        = 10^-8                #minimal singular value of the matrix
    tol         = 10^-11               #tolerance

    # CREATE A MATRIX
    smax = np.log(smax)
    smin = np.log(smin)
    kk = n - k
    a = -(smax - smin) / (kk - 1)
    S = np.arange(smax, smin, a)
    S = diags(np.concatenate((np.exp(S), np.zeros(k))), format='csc')

    U = eye(n).tocsc()
    dU = random(n, n, density=1/n*K, format='csc', data_rvs=np.random.randn)
    U = U + dU
    P = np.random.permutation(n)
    U = U[P, :]
    A = U @ S @ inv(U)

    #  DISPLAY PARAMETERS
    print(' ')
    print('The Matrix:')
    print(f'  {n} - dimension')
    print(f'  {k} - null space dimension')
    print(f'  {np.count_nonzero(A)} - number of nonzero elements')
    print(' ')
    print(f'  {np.exp(smax)} - maximal singular value')
    print(f'  {np.exp(smin)} - minimal singular value')
    print(f'  {tol} - tolerance')
    print(' ')

    #  FIND NULL SPACE
    start_time = time.time()
    SpLeft = spspaces(A, 1, 10**-10)  
    t = time.time() - start_time

    Q = SpLeft[0]
    J = SpLeft[2]

    kL = len(J)
    nL = np.linalg.norm(Q[J, :] @ A, 1)


    # DISPLAY RESULTS
    print('Results:')
    print(f'   {t}  - elapsed time')
    print('    {kL} - estimated null space dimension')
    print('    {nL} - norm of residuals N*A, where N is a null space')
