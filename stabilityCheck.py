from solvers import solve_1d_wave_eq_semi_forward
import numpy as np

import matplotlib.pyplot as plt

@np.vectorize
def is_stable(N_x, iterations, L, T, c):
    dx = L / (N_x + 1)
    dt = T / iterations

    mu = dt * (c / dx) ** 2

    U,xd,td = solve_1d_wave_eq_semi_forward(
        g=lambda x: x*(L-x),
        f=lambda x: 0 * x,
        a=lambda t: 0 * t,
        b=lambda t: 0 * t,
        N_x=N_x,
        iterations=iterations,
        L=L,
        T=T,
        c=c
    )

    if np.max(U.T[N_x//2]) > L**2:
        return False, mu
    
    return True, mu


@np.vectorize
def is_stable_dt_dx(dx, dt, L, T, c):
    N_x = int(L / dx) - 1
    iterations = int(T / dt)
  

    mu = dt * (c / dx) ** 2

    U,xd,td = solve_1d_wave_eq_semi_forward(
        g=lambda x: x*(L-x),
        f=lambda x: 0 * x,
        a=lambda t: 0 * t,
        b=lambda t: 0 * t,
        N_x=N_x,
        iterations=iterations,
        L=L,
        T=T,
        c=c
    )

    if np.max(U.T[N_x//2]) > L**2:
        return False, mu
    
    return True, mu


if __name__ == "__main__":
    L = 4
    T = 4
    c = 1


    Dx = np.linspace(0.001, 1, 100)
    Dt = np.linspace(0.001, 1, 100)

    Y, X = np.meshgrid(Dt, Dx,indexing='ij')

    region = is_stable_dt_dx(dx=X,
        dt=Y,
        L=L,
        T=T,
        c=c
    )[0]

    plt.figure(figsize=(6, 6))
    plt.contourf(X, Y, region, levels=[0.5,1], colors=["lightblue"])
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.show()




    
   

  

