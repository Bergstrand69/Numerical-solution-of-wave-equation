import numpy as np

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

def solve_2d_wave_eq_semi_forward(U0, V0, Ua, Ub, Uc, Ud, N_x, N_y, iterations, L_x, L_y, T, c=1, alpha=0.5):
    xi = np.linspace(0,L_x,N_x+2)
    yi = np.linspace(0,L_y,N_y+2)
    ti = np.linspace(0,T,iterations+1)
    U = np.zeros((iterations+1, N_x+2, N_y+2))
    V = np.zeros((iterations+1, N_x, N_y))

    t_x, x  = np.meshgrid(ti, xi,indexing='ij')
    t_y, y  = np.meshgrid(ti, yi,indexing='ij')
   

    U[:,0, :] = Ua(t_y,y)
    U[:, :, 0] = Ub(t_x, x)
    U[:, -1, :] = Uc(t_y, y)
    U[:, :, -1] = Ud(t_x, x)

    print(len(xi), len(yi))

    
    grid = np.meshgrid(xi, yi,indexing='ij')

    print(grid[0].shape, grid[1].shape)

    U[0, 1:-1, 1:-1] = U0(grid[0][1:-1, 1:-1], grid[1][1:-1, 1:-1])
    V[0, :, :] = V0(grid[0][1:-1, 1:-1], grid[1][1:-1, 1:-1])

    dx = L_x / (N_x + 1)
    dy = L_y / (N_y + 1)
    dt = T / iterations

    for j in range(iterations):
        V[j+1] = V[j] +  dt * c**2 * (1/dx**2 * (U[j,2:,1:-1] + U[j,:-2,1:-1] - 2*U[j,1:-1,1:-1]) \
                + 1/dy**2 * (U[j,1:-1,2:] + U[j,1:-1,:-2] - 2*U[j,1:-1,1:-1])) - alpha * V[j] * dt
        U[j+1,1:-1, 1:-1] = U[j,1:-1, 1:-1] + dt*V[j+1]

    return U, grid, ti


if __name__ == "__main__":
    L_x = 3
    L_y = 6
    T = 50
    c = 1
    N_x = 50
    N_y = 50
    iterations = 1000

    U, grid, ti = solve_2d_wave_eq_semi_forward(
        U0=lambda x,y: 0*np.sin(np.pi * x/L_x) * np.sin(np.pi * y/L_y),
        V0=lambda x,y: np.zeros_like(y),
        Ua=lambda t, y: np.sin(3*np.pi * t/L_x)*np.sin(np.pi * y/L_y),
        Ub=lambda t,x: np.zeros_like(x),
        Uc=lambda t, x: np.zeros_like(x),
        Ud=lambda t, y: np.zeros_like(y),
        N_x=N_x,
        N_y=N_y,
        iterations=iterations,
        L_x=L_x,
        L_y=L_y,
        T=T,
        c=c,
        alpha = 1
    )


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim(0, L_x)
    ax.set_ylim(0, L_y)
    ax.set_zlim(-1, 1)

    X, Y = grid

    surface = [ax.plot_surface(X, Y, U[0], color='b')]

   

    
    
    def update(frame):
        surface[0].remove()
        surface[0] = ax.plot_surface(X, Y, U[frame], color='b', alpha=0.8)
        return surface
        

    frameRateRaduction = 1
    ani = FuncAnimation(fig,update, frames=range(0, iterations, frameRateRaduction), interval=frameRateRaduction*1000*T/iterations, blit=False)
    ani.save("Numerical-solution-of-wave-equation\Resonance_example2D.gif")

    plt.show()