import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from utils import roll, pitch, yaw

plt.ion()  # enable interactive drawing


class dataPlotter:
    '''
        This class plots the time histories for the ACC data.
    '''

    def __init__(self):
        # Number of subplots = num_of_rows*num_of_cols
        self.num_rows = 4    # Number of subplot rows
        self.num_cols = 1    # Number of subplot columns

        # Crete figure and axes handles
        self.fig, self.ax = plt.subplots(self.num_rows, self.num_cols, sharex=True)

        # Instantiate lists to hold the time and data histories
        self.time_history = []  # time
        self.x_history = []  # velocity
        self.y_history = []  # velocity
        self.z_history = []  # velocity
        self.vx_history = []  # velocity
        self.vy_history = []  # velocity
        self.vz_history = []  # velocity
        self.u1_history = []  # velocity reference
        self.u2_history = []  # velocity reference
        self.u3_history = []  # velocity reference
        self.u4_history = []  # velocity reference
        self.phi_history = []
        self.theta_history = []
        self.psi_history = []

        # create a handle for every subplot.
        self.handle = []
        self.handle.append(myPlot(self.ax[0], ylabel='pos', title='Data'))
        self.handle.append(myPlot(self.ax[1], ylabel='vel', title='Data'))
        # self.handle.append(myPlot(self.ax[1], ylabel='x2'))
        self.handle.append(myPlot(self.ax[2], ylabel='angles', xlabel='t(s)'))
        self.handle.append(myPlot(self.ax[3], ylabel='u', xlabel='t(s)'))
        # self.handle.append(myPlot(self.ax[3], ylabel='slack'))
        # self.handle.append(myPlot(self.ax[4], ylabel='BLF(V)'))
        # self.handle.append(myPlot(self.ax[5], xlabel='t(s)', ylabel='CBF(B)'))

    def update(self, t, states, ctrl):
        '''
            Add to the time and data histories, and update the plots.
        '''
        # update the time history of all plot variables
        self.time_history.append(t)  # time
        self.x_history.append(states[0,0])
        self.y_history.append(states[1,0])
        self.z_history.append(states[2,0])
        self.vx_history.append(states[3,0])
        self.vy_history.append(states[4,0])
        self.vz_history.append(states[5,0])
        self.u1_history.append(ctrl[0,0])
        self.u2_history.append(ctrl[1,0])
        self.u3_history.append(ctrl[2,0])
        self.u4_history.append(ctrl[3,0])

        q = states[6:10]

        self.phi_history.append(roll(q))
        self.theta_history.append(pitch(q))
        self.psi_history.append(yaw(q))


        # self.p00_history.append(P[0,0])
        # self.p01_history.append(P[0,1])
        # self.p10_history.append(P[1,0])
        # self.p11_history.append(P[1,1])
        # self.slack_history.append(slack)
        # self.V_history.append(V)
        # self.B_history.append(B)

        # update the plots with associated histories
        self.handle[0].update(self.time_history, [self.x_history, self.y_history, self.z_history])
        self.handle[1].update(self.time_history, [self.vx_history, self.vy_history, self.vz_history])
        self.handle[2].update(self.time_history, [self.phi_history, self.theta_history,
                                                    self.psi_history])
        self.handle[3].update(self.time_history, [self.u1_history, self.u2_history, self.u3_history, self.u4_history])

    def show(self):
        self.fig.show()

class myPlot:
    '''
        Create each individual subplot.
    '''
    def __init__(self, ax,
                 xlabel='',
                 ylabel='',
                 title='',
                 legend=None):
        '''
            ax - This is a handle to the  axes of the figure
            xlable - Label of the x-axis
            ylable - Label of the y-axis
            title - Plot title
            legend - A tuple of strings that identify the data.
                     EX: ("data1","data2", ... , "dataN")
        '''
        self.legend = legend
        self.ax = ax                  # Axes handle
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'b']
        # A list of colors. The first color in the list corresponds
        # to the first line object, etc.
        # 'b' - blue, 'g' - green, 'r' - red, 'c' - cyan, 'm' - magenta
        # 'y' - yellow, 'k' - black
        self.line_styles = ['-', '-', '--', '-.', ':']
        # A list of line styles.  The first line style in the list
        # corresponds to the first line object.
        # '-' solid, '--' dashed, '-.' dash_dot, ':' dotted

        self.line = []

        # Configure the axes
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel(xlabel)
        self.ax.set_title(title)
        self.ax.grid(True)

        # Keeps track of initialization
        self.init = True

    def update(self, time, data):
        '''
            Adds data to the plot.
            time is a list,
            data is a list of lists, each list corresponding to a line on the plot
        '''
        if self.init == True:  # Initialize the plot the first time routine is called
            for i in range(len(data)):
                # Instantiate line object and add it to the axes
                self.line.append(Line2D(time,
                                        data[i],
                                        color=self.colors[np.mod(i, len(self.colors) - 1)],
                                        ls=self.line_styles[np.mod(i, len(self.line_styles) - 1)],
                                        label=self.legend if self.legend != None else None))
                self.ax.add_line(self.line[i])
            self.init = False
            # add legend if one is specified
            if self.legend != None:
                plt.legend(handles=self.line)
        else: # Add new data to the plot
            # Updates the x and y data of each line.
            for i in range(len(self.line)):
                self.line[i].set_xdata(time)
                self.line[i].set_ydata(data[i])

        # Adjusts the axis to fit all of the data
        self.ax.relim()
        self.ax.autoscale()
        plt.draw()

    # def show(self):
    #     plt.show()
