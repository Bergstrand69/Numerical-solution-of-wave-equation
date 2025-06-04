import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animateSimulation2d(U,xd,iterations,T, x_lim = [0,1], y_lim = [-1,1], save = False, frameRateRaduction = 1, show = True, title = "",path = "Numerical-solution-of-wave-equation"):
    
    fig = plt.figure()
    ax = fig.add_subplot()
    line = ax.plot([],[],lw=1)[0]
    
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    def update(frame):
        line.set_data(xd,U[frame])
        return [line]

    ani = FuncAnimation(fig,
                        update,
                        frames = range(0,iterations,frameRateRaduction),
                        interval = frameRateRaduction*1000*T/iterations,
                        blit = True
                        )
    plt.title(title)
    ani.save(path +"\\" + title + ".gif") if save else None

    if show:
        plt.show()