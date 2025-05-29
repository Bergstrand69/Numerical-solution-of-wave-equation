import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

def solve_2d_wave_eq_semi_forward(U0, V0, Ua, Ub, Uc, Ud, N_x, N_y, iterations, L_x, L_y, T, c=1):
    xi = np.linspace(0,L_x,N_x+2)
    yi = np.linspace(0,L_y,N_y+2)
    ti = np.linspace(0,T,iterations+1)
    U = np.zeros((iterations+1, N_x+2, N_y+2))
    V = np.zeros((iterations+1, N_x, N_y))
    U[0,0, :] = Ua(yi)
    U[0, :, 0] = Ub(xi)
    U[0, -1, :] = Uc(xi)
    U[0, :, -1] = Ud(yi)
    
    grid = np.meshgrid(xi, yi)
    print(grid)
    U[0, 1:-1, 1:-1] = U0(grid[0][1:-1, 1:-1], grid[1][1:-1, 1:-1])
    V[0, :, :] = V0(grid[0][1:-1, 1:-1], grid[1][1:-1, 1:-1])

    dx = L_x / (N_x + 1)
    dy = L_y / (N_y + 1)
    dt = T / iterations

    for j in range(iterations):

        V[j+1] = V[j] + c**2 * (1/dx**2 * (U[j,2:,1:-1] + U[j,:-2,1:-1] - 2*U[j,1:-1,1:-1]) \
                + 1/dy**2 * (U[j,1:-1,2:] + U[j,1:-1,:-2] - 2*U[j,1:-1,1:-1]))
        U[j+1,1:-1, 1:-1] = U[j,1:-1, 1:-1] + dt*V[j+1]
    
    return U, grid, ti


if __name__ == "__main__":
    L_x = 3
    L_y = 3
    T = 10
    c = 2
    N_x = 10
    N_y = 10
    iterations = 2000

    U, grid, ti = solve_2d_wave_eq_semi_forward(
        U0=lambda x,y: np.sin(np.pi * x) * np.sin(np.pi * y),
        V0=lambda x,y: np.zeros_like(y),
        Ua=lambda y: np.zeros_like(y),
        Ub=lambda x: np.zeros_like(x),
        Uc=lambda x: np.zeros_like(x),
        Ud=lambda y: np.zeros_like(y),
        N_x=N_x,
        N_y=N_y,
        iterations=iterations,
        L_x=L_x,
        L_y=L_y,
        T=T,
        c=c
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = grid
    ax.plot_surface(X, Y, U[5], cmap='viridis')

    plt.show()