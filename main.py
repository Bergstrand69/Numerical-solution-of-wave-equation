from solvers import solve_1d_wave_eq_forward,solve_1d_wave_eq_backward,solve_1d_wave_eq_backward_energy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
if __name__ == "__main__":
    L = 3
    T = 10
    c = 20
    N_x = 20
    iterations = 2000

    U,xd,td =solve_1d_wave_eq_forward(
        g = lambda x : 0*x*(L-x) ,
        f = lambda x : 0*x,
        a = lambda t : 0*t,
        b = lambda t : np.sin(3*c*np.pi*t/L),
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

    frameRateRaduction = 1

    ani = FuncAnimation(fig,
                        update,
                        frames = range(0,iterations,frameRateRaduction),
                        interval = frameRateRaduction*1000*T/iterations,
                        blit = True
                        )
    plt.show()

    #ani.save("Numerical-solution-of-wave-equation\Resonance_example.gif")

    plt.plot(td,U.T[N_x//2])

    plt.show()