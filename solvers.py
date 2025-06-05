import numpy as np


def solve_1d_wave_eq_semi_forward(g,f,a,b,N_x,iterations,L, T,c = 1):
    xi = np.linspace(0,L,N_x+2)
    ti = np.linspace(0,T,iterations+1)
    W = np.zeros((iterations+1,N_x))
    U = np.zeros((iterations+1,N_x+2))
    W[0] = f(xi[1:-1])
    U[0,1:-1] = g(xi[1:-1])
    U[:,0] = a(ti)
    U[:,N_x+1] = b(ti)

    dx = L/(N_x+1)
    dt = T/iterations
    
    k = dt * (c/dx)**2

    for j in range(iterations):
        W[j+1] = W[j] + k*(U[j,2:] + U[j,:-2] -2*U[j,1:-1])
        U[j+1,1:-1] = U[j,1:-1] + dt*W[j+1]

    return U,xi,ti

def solve_1d_wave_eq_semi_forward_no_boundary(g,f,h,N_x,iterations,L, T,c = 1, k = 0, alpha = 0, ro = 1):
    """
    Solves the 1D wave equation using a semi-forward method, with no boundrary.

    Parameters:
        g : function
            Initial displacement as a function of x.
        f : function
            Initial velocity as a function of x.
        h : function 
            Right hands side of the eqation, functrion of x and t.
        N_x : int
            Number of spatial grid points (excluding boundaries).
        iterations : int
            Number of time steps.
        L : float
            Length of the spatial domain.
        T : float
            Total simulation time.
        c : float, optional
            Wave speed (default is 1).
        k : float, optional
            spring constant (default is 0).
        alpha : float, optional
            Damping coefficient (default is 0).
        ro : float, optional
            Density of medium (default is 1).
    Returns:
        U : ndarray
            Solution array of shape (iterations+1, N_x+2).
        xi : ndarray
            Spatial grid points.
        ti : ndarray
            Time grid points.
    """

    xi = np.linspace(0,L,N_x+2)
    ti = np.linspace(0,T,iterations+1)
    W = np.zeros((iterations+1,N_x+2))
    U = np.zeros((iterations+1,N_x+2))
    W[0] = f(xi)
    U[0] = g(xi)
    
    dx = L/(N_x+1)
    dt = T/iterations
    
    r = dt * (c/dx)**2

    for j in range(iterations):
        W[j+1,1:-1] = W[j,1:-1] + r*(U[j,2:] + U[j,:-2] -2*U[j,1:-1]) + h(xi[1:-1], ti[j])*dt / ro -  k*U[j,1:-1]/c**2 * dt / ro - alpha * W[j,1:-1] * dt / ro 
        W[j+1,-1] = -c*(U[j,-1] - U[j,-2])/(dx) - k*U[j,-1]/c**2 * dt / ro - alpha * W[j,-1] * dt / ro
        W[j+1,0 ] = -c*(U[j,-1] - U[j,-2])/(dx) - k*U[j,0 ]/c**2 * dt / ro - alpha * W[j,0] * dt / ro
        U[j+1] = U[j] + dt*W[j+1] 
        

    return U,xi,ti
   

def solve_1d_wave_eq_backward(g,f,a,b,N_x,iterations,L, T,c = 1):
    xi = np.linspace(0,L,N_x+2)
    ti = np.linspace(0,T,iterations+1)
    W = np.zeros((iterations+1,N_x))
    U = np.zeros((iterations+1,N_x+2))
    W[0] = f(xi[1:-1])
    U[0,1:-1] = g(xi[1:-1])
    U[:,0] = a(ti)
    U[:,N_x+1] = b(ti)
    
    dx = L/(N_x+1)
    dt = T/iterations
    
    k = dt * (c/dx)**2

    for j in range(iterations):
        
        w = W[j] + k*(U[j,2:] + U[j,:-2] -2*U[j,1:-1])
        
        u = U[j]
        u[1:-1] = U[j,1:-1] + dt*W[j]
        
        
        for _ in range(10): 
            temp_w = W[j] + k*(u[2:] + u[:-2] -2*u[1:-1])
            u[1:-1] = U[j,1:-1] + dt*w
            w = temp_w
        
        W[j+1] = w
        U[j+1,1:-1] = u[1:-1]
        

    return U,xi,ti

def solve_1d_wave_eq_backward_energy(g,f,a,b,N_x,iterations,L, T,c = 1):
    xi = np.linspace(0,L,N_x+2)
    ti = np.linspace(0,T,iterations+1)
    W = np.zeros((iterations+1,N_x))
    U = np.zeros((iterations+1,N_x+2))
    W[0] = f(xi[1:-1])
    U[0,1:-1] = g(xi[1:-1])
    U[:,0] = a(ti)
    U[:,N_x+1] = b(ti)
    
    dx = L/(N_x+1)
    dt = T/iterations
    
    k = dt * (c/dx)**2

    E0 = getEnergy(U[0],W[0])
    
    for j in range(iterations):
        
        w = W[j] + k*(U[j,2:] + U[j,:-2] -2*U[j,1:-1])
        
        u = U[j]
        u[1:-1] = U[j,1:-1] + dt*W[j]
        
        
        for _ in range(10): 
            w = W[j] + k*(u[2:] + u[:-2] -2*u[1:-1])
            u[1:-1] = U[j,1:-1] + dt*w
        
        E_tot = getEnergy(u,w)
        W[j+1] = w*np.sqrt(E0/E_tot)
        U[j+1,1:-1] = u[1:-1]*np.sqrt(E0/E_tot)
        

    return U,xi,ti

def getEnergy(y,v):
    return np.sum(y**2) + np.sum(v**2)