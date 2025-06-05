from solvers import solve_1d_wave_eq_semi_forward,solve_1d_wave_eq_backward,solve_1d_wave_eq_backward_energy,solve_1d_wave_eq_semi_forward_no_boundary
from solvers_heat_eq import solve_1d_heat_eq_forward
from stabilityCheck import is_stable

from Ploter import animateSimulation2d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
if __name__ == "__main__":
    L = 10
    T = 10
    N_x = 80
    iterations = 1000

    dt = T/iterations
    dx = L/(N_x+1)
    
    c = 2

    def U(x):
        return np.piecewise(x, [x < 0, x >= 0], [lambda x: 0*x, lambda x: 1 + 0*x])

    
    U,xd,td =solve_1d_wave_eq_semi_forward_no_boundary(
        g = lambda x : 0*np.exp(-((x-L/2)**2)/0.5**2), 
        f = lambda x : 0*x,
        h = lambda x, t : 40*np.exp(-((x-L/2)**2)/0.5**2)*(U(t) - U(t-0.1) + U(t- 2) - U(t-2.1)),
        N_x = N_x,
        iterations = iterations,
        L = L,
        T = T,
        c = c,
        alpha= 0.5,
        k = 10,
        ro = 1,
        )
    
    

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