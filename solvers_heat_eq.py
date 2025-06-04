import numpy as np



def solve_1d_heat_eq_forward(g,a,b,N_x,iterations,L, T,alpha = 1):
    xi = np.linspace(0,L,N_x+2)
    ti = np.linspace(0,T,iterations+1)
    U = np.zeros((iterations+1,N_x+2))
    
    U[0,1:-1] = g(xi[1:-1])
    U[:,0] = a(ti)
    U[:,N_x+1] = b(ti)

    dx = L/(N_x+1)
    dt = T/iterations

    for j in range(iterations):
        U[j+1,1:-1] = U[j,1:-1] + alpha*(U[j,2:] + U[j,:-2] -2*U[j,1:-1]) * dt / dx**2
    
    return U,xi,ti