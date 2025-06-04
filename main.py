from solvers import solve_1d_wave_eq_semi_forward,solve_1d_wave_eq_backward,solve_1d_wave_eq_backward_energy
from solvers_heat_eq import solve_1d_heat_eq_forward
from stabilityCheck import is_stable

from Ploter import animateSimulation2d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
if __name__ == "__main__":
    L = 4
    T = 10
    N_x = 1600
    iterations = 100

    dt = T/iterations
    dx = L/(N_x+1)
    
    c = 1



    
    U,xd,td =solve_1d_wave_eq_semi_forward(
        g = lambda x : 4*np.exp(-((x-L/2)**2)/0.5**2), 
        f = lambda x : 0*x,
        a = lambda t : 0*np.piecewise(t, [t < 1, t >= 1], [lambda t: 0*t, lambda t: 1 + 0*t]),
        b = lambda t : 0*t,
        N_x = N_x,
        iterations = iterations,
        L = L,
        T = T,
        c = c
        )
    
    print(is_stable(N_x, iterations, L, T, c))

    animateSimulation2d(
        U = U,
        xd = xd,
        iterations = iterations,
        T = T,
        x_lim = [0,L],
        y_lim = [-5,5],
        save = False,
        frameRateRaduction = 10,
        show = True,
        title="1D Wave Equation Solution"
    )
  
   

    

    #plt.plot(td,U.T[N_x//2])

    plt.show()