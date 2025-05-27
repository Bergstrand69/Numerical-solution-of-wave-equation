from solvers import solve_1d_wave_eq_forward,solve_1d_wave_eq_backward,solve_1d_wave_eq_backward_energy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
if __name__ == "__main__":
    L = 3
    T = 5
    c = 1
    N_x = 40*5
    iterations = 500

    U,xd,td =solve_1d_wave_eq_backward(
        g = lambda x : 0*x ,
        f = lambda x : 0*x,
        a = lambda t : 0*t,
        b = lambda t : np.sin(4*np.pi*c*t/L),
        N_x = N_x,
        iterations = iterations,
        L = L,
        T = T,
        c = c
        )
  
    print(U)
    
    fig = plt.figure()
    ax = fig.add_subplot()
    line = ax.plot([],[],lw=1)[0]
    
    ax.set_xlim(0,L)
    ax.set_ylim(-5,5)

    def update(frame):
        line.set_data(xd,U[frame])
        return [line]

    frameRateRaduction = 7

    ani = FuncAnimation(fig,
                        update,
                        frames = range(0,iterations,frameRateRaduction),
                        interval = frameRateRaduction*1000*T/iterations,
                        blit = True
                        )
    plt.show()

    ani.save("Numerical-solution-of-wave-equation\Resonance_example.gif")

    plt.plot(td,U.T[N_x//2])

    plt.show()