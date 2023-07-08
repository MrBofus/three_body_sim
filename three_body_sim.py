##````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````##
# Created by: Me :)
# 
# Created on 07-02-2023
# Purpose: Three-body (really n-body) simulation with drag + speed limits and whatnot


##````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````##
# import libraries
#
# libraries imported:
#   matplotlib  -   used to make the figures
#   pandas      -   used to store variables for plotting + analysis purposes
#   numpy       -   python fast math library
#   random      -   used to initialize the variables to be random each time

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import random


##````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````##
# define general-purpose functions
#
# these functions are used to build larger composite functions


##-----------------------------------
# function: magnitude
#
# accepts: 
#   array or array-like object
#
# returns: 
#   float magnitude of vector

def magnitude(vec):
    # initialize magnitude at zero
    magnitude = 0

    # loop through each element in array,
    # summing the squares of the elements
    for v in vec:
        magnitude += v**2
    
    # return the square root of the sum, resulting 
    # in the magnitude of the array
    return np.sqrt(magnitude)


##-----------------------------------
# function: normalize
#
# accepts: 
#   array or array-like object
#
# returns: 
#   array of magnitude 1

def normalize(vec):
    # determine magnitude of vector
    m = magnitude(vec)
    
    # initialize return vector
    # as empty array
    v_ = []
    for v in vec:
        # append each element of input vector
        # divided by magnitude
        v_.append(v / m)
    
    # return normalized vector
    return np.array(v_)


##-----------------------------------
# function: check radius
#
# accepts: 
#   array or array-like object (location of point 1)
#   array or array-like object (location of point 2)
#   distance threshold
#
# returns: 
#   boolean: two points are not within radius
#            r of each other or they are

def check_radius(r1, r2, r):
    # initialize distance magntidue at zero
    i_ = 0

    # sum the squares of the distance
    # between each point per axis
    for i in range(len(r1)):
        i_ += (r1[i] - r2[i]) ** 2

    # take the square root of the sum,
    # resulting in the magnitude of 
    # distance between the two points
    radius = np.sqrt(i_)

    # if the distance is greater
    # than threshold, return true
    if radius > r:
        return True
    
    # if they are within the 
    # threshold, return false
    else:
        return False


##````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````##
# define higher-layer functions
#
# these functions are used to run the simulation


##-----------------------------------
# function: compute acceleration
#
# accepts: 
#   array containing the locations of each of the bodies
#   constant determining strength of gravity
#
# returns: 
#   array containing the x-, y-, and z- components
#       of the acceleration acting on the body due to gravity
'''
def compute_acceleration(rvec, k):
    # initialize the x-, y-, and z- components of accleration
    # at zero
    ax = 0
    ay = 0
    az = 0


    for i in range(1, len(rvec)):
        dist_x = rvec[0][0] - rvec[i][0]
        dist_y = rvec[0][1] - rvec[i][1]
        dist_z = rvec[0][2] - rvec[i][2]

        i_ = dist_x**2 + dist_y**2 + dist_z**2
        radii = np.sqrt(i_)**3
        
        ax += -k * dist_x/radii
        ay += -k * dist_y/radii
        az += -k * dist_z/radii

    return np.array([ax, ay, az])
'''

##-----------------------------------
# function: drag term
#
# accepts: 
#   constant determining strength of drag
#   vector represting velocity
#
# returns: 
#   array containing the x-, y-, and z- components
#       of the acceleration acting on the body due to drag

def drag_term(C, v):
    v_direction = normalize(v)
    
    return -C * v_direction * magnitude(v)**2


def integrate_acceleration(a, v, timestep, speed_limit):

    v_x = v[0] + a[0] * timestep
    v_y = v[1] + a[1] * timestep
    v_z = v[2] + a[2] * timestep
    
    if magnitude([v_x, v_y, v_z]) > speed_limit:
        v_ = speed_limit*normalize([v_x, v_y, v_z])
    
        v_x = v_[0]
        v_y = v_[1]
        v_z = v_[2]

    return np.array([v_x, v_y, v_z])


def integrate_velocity(v, r, timestep):
    x = r[0] + v[0] * timestep
    y = r[1] + v[1] * timestep
    z = r[2] + v[2] * timestep
    
    return np.array([x, y, z])


def run_check(bodies, dist):
    for i in range(len(bodies)):
        for j in range(len(bodies)):
            if not i == j:
                if check_radius(bodies[j].r, bodies[i].r, dist):
                    return True
    
    return False
                        
                        

class body():
    def __init__(self, a, r, v, m, G, C, drag_term, timestep, speed_limit):
        self.a = a
        self.r = r
        self.v = v
        
        self.Fg = np.array([0, 0, 0])
        self.Fd = np.array([0, 0, 0])

        self.G = G
        self.m = m
        self.timestep = timestep
        
        self.C = C
        self.drag = drag_term
        
        self.speed_limit = speed_limit
    
    def update_acceleration(self, r_vec, m_vec):
        
        ax = 0
        ay = 0
        az = 0
        for i in range(0, len(rvec)):
            dist_x = self.r[0] - rvec[i][0]
            dist_y = self.r[1] - rvec[i][1]
            dist_z = self.r[2] - rvec[i][2]

            i_ = dist_x**2 + dist_y**2 + dist_z**2
            radii = np.sqrt(i_)**3

            if not ( abs(dist_x) < 0.1 ):
                ax += -(self.G * m_vec[i] * dist_x/radii) # / self.m
            
            if not ( abs(dist_y) < 0.1 ):
                ay += -(self.G * m_vec[i] * dist_y/radii) # / self.m
            
            if not ( abs(dist_z) < 0.1 ):
                az += -(self.G * m_vec[i] * dist_z/radii) # / self.m
            
        self.Fg = self.m*np.array([ax, ay, az])

        if self.drag == True:
            F_drag = drag_term(self.C, self.v)
            self.Fd = self.m*F_drag

            ax += F_drag[0]
            ay += F_drag[1]
            az += F_drag[2]

        self.a = np.array([ax, ay, az])
    
    def update_position(self):
        vel = integrate_acceleration(self.a, self.v, self.timestep, self.speed_limit)
        rvec = integrate_velocity(self.v, self.r, self.timestep)
        
        self.v = vel
        self.r = rvec
        

def colorMap(c1, c2, n):
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-n)*c1 + n*c2)


def interpolate_between_colors(color1, color2, i):
    return i*np.array(color1) + (1-i)*np.array(color2)


def plot_with_gradient(axes, x, y, z, t,
                       gap):

    for i in range(1, len(x), gap):
        iterator = i/len(x)

        colorcode = interpolate_between_colors([0.9999, 0.9999, 0.9999], 
                                               [0.0000, 0.0000, 0.0000], iterator)
        
        axes.plot(x[i-1:i+gap], y[i-1:i+gap], z[i-1:i+gap], c=colorcode)



##````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````##



timestep = 0.1
    
distance = 5
speed = 1/10
disp = 1

k = 0.5 # * 0.01

C = 0.2
include_drag = True

three_dimension_projection = 1
three_dimensional_plot = True

show_forces = False

speed_limit = 2

nbodies = random.randint(2, 5)

def mfunction(i):
    return 1


##````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````##

if three_dimensional_plot:
    fig = plt.figure(facecolor='black')
    ax = plt.axes(projection='3d')
else:
    fig, ax = plt.subplots(facecolor='black') 

fig.canvas.toolbar.pack_forget()
plt.rcParams['toolbar'] = 'None' # Remove tool bar (upper bar)
# fig.canvas.window().statusBar().setVisible(False) # Remove status bar (bottom bar)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

##````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````##
    
while True:
    t = 0
    

    ##````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````##
    m = []
    for i in range(nbodies):
        m.append(mfunction(i))

    ##````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````##
    
    v_to_start = []
    for i in range(nbodies):
        v_to_start.append( (speed/m[i]) * np.array([1-random.random(), 1-random.random(), three_dimension_projection*(1-random.random())]) )
    
    angle_degrees = 360/nbodies
    angle = angle_degrees*np.pi/180
    cosine = np.cos(angle)
    sine = np.sin(angle)
    
    r_to_start = [ distance*normalize([1-random.random(), 1-random.random(), three_dimension_projection*(1-random.random())]) ]
    for i in range(1, nbodies):
        r_to_start.append( np.array([r_to_start[i-1][0]*cosine - r_to_start[i-1][1]*sine + disp*(1-random.random()),
                                     r_to_start[i-1][0]*sine + r_to_start[i-1][1]*cosine + disp*(1-random.random()),
                                     three_dimension_projection*(1-random.random())/10]) )
    
    ##````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````##
    
    dataframes = []
    bodies = []
    for i in range(nbodies):
        bodies.append(body([0, 0, 0], r_to_start[i], v_to_start[i], m[i], k, C, include_drag, timestep, speed_limit))
        dataframes.append( pd.DataFrame({'time': [0], 
                                         'x': [bodies[i].r[0]], 'y': [bodies[i].r[1]], 'z': [bodies[i].r[2]],
                                         'vx': [bodies[i].v[0]], 'vy': [bodies[i].v[1]], 'vz':[bodies[i].v[2]],
                                         'Fgx':[bodies[i].Fg[0]], 'Fgy':[bodies[i].Fg[1]], 'Fgz':[bodies[i].Fg[2]],
                                         'Fdx':[bodies[i].Fd[0]], 'Fdy':[bodies[i].Fd[1]], 'Fdz':[bodies[i].Fd[2]]}))
    
    ##````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````##

    while True:
    
        ax.clear()
        
        ##````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````##

        rvec = []
        mvec = []
        for i in range(len(bodies)):
            rvec.append(bodies[i].r)
            mvec.append(bodies[i].m)
        
        for i in range(len(bodies)):
            bodies[i].update_acceleration(rvec, mvec)
        
        ##````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````##

        for i in range(len(bodies)):
            bodies[i].update_position()
            tempdf = pd.DataFrame({'time': [t], 
                                    'x': [bodies[i].r[0]], 'y': [bodies[i].r[1]], 'z': [bodies[i].r[2]],
                                    'vx': [bodies[i].v[0]], 'vy': [bodies[i].v[1]], 'vz':[bodies[i].v[2]],
                                    'Fx':[bodies[i].a[0]], 'Fy':[bodies[i].a[1]], 'Fz':[bodies[i].a[2]],
                                    'Fgx':[bodies[i].Fg[0]], 'Fgy':[bodies[i].Fg[1]], 'Fgz':[bodies[i].Fg[2]],
                                    'Fdx':[bodies[i].Fd[0]], 'Fdy':[bodies[i].Fd[1]], 'Fdz':[bodies[i].Fd[2]]})
            dataframes[i] = pd.concat( [dataframes[i], tempdf] )
        
        ##````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````##
        
        for i in range(len(bodies)):
            
            if not three_dimensional_plot:
                ax.plot( np.array(dataframes[i]['x'].tail(500)), np.array(dataframes[i]['y'].tail(500)), 
                        color='white')
                '''
                if len(dataframes[i]) > 550:
                    for j in range(len(dataframes[i]), len(dataframes[i]) + 50, 5):
                        ax.scatter( dataframes[i]['x'].iloc[j], dataframes[i]['y'].iloc[j],
                                color=colorMap('black', 'white', j/50), s=0.1)
                '''
                '''
                if len(dataframes[i]) < 100:
                    for j in range(1, len(dataframes[i])):
                        xrange = np.array([ dataframes[i]['x'].iloc[j - 1], dataframes[i]['x'].iloc[j] ])
                        yrange = np.array([ dataframes[i]['y'].iloc[j - 1], dataframes[i]['y'].iloc[j] ])
                        ax.plot( xrange, yrange, color=colorMap('black', 'white', j/len(dataframes[i])))
                else:
                    for j in range(len(dataframes[i])-100, len(dataframes[i])):
                        xrange = np.array([ dataframes[i]['x'].iloc[j - 1], dataframes[i]['x'].iloc[j] ])
                        yrange = np.array([ dataframes[i]['y'].iloc[j - 1], dataframes[i]['y'].iloc[j] ])
                        ax.plot( xrange, yrange, color=colorMap('black', 'white', j/len(dataframes[i])))
                '''
                
                ax.scatter( np.array(dataframes[i]['x'].tail(1)), np.array(dataframes[i]['y'].tail(1)), 
                            color='white', s=0.1*bodies[i].m**2 )
                
                if show_forces == True:
                    r_start = np.array([np.array(dataframes[i]['x'].tail(1)), np.array(dataframes[i]['y'].tail(1))])

                    r_end_g =  np.array([np.array(dataframes[i]['Fgx'].tail(1)), np.array(dataframes[i]['Fgy'].tail(1))])
                    ax.quiver(r_start[0], r_start[1],    r_end_g[0], r_end_g[1],     color='red', width=0.001, zorder=100)

                    r_end_d =  np.array([np.array(dataframes[i]['Fdx'].tail(1)), np.array(dataframes[i]['Fdy'].tail(1))])
                    ax.quiver(r_start[0], r_start[1],    r_end_d[0], r_end_d[1],     color='blue', width=0.001, zorder=100)

                    r_end_t =  np.array([np.array(dataframes[i]['Fdx'].tail(1))+np.array(dataframes[i]['Fgx'].tail(1)), 
                                        np.array(dataframes[i]['Fdy'].tail(1))+np.array(dataframes[i]['Fgy'].tail(1))])
                    ax.quiver(r_start[0], r_start[1],    r_end_t[0], r_end_t[1],     color='green', width=0.001, zorder=100)
            
            else:
                ax.plot( np.array(dataframes[i]['x'].tail(500)), np.array(dataframes[i]['y'].tail(500)), np.array(dataframes[i]['z'].tail(500)),
                        color='white')
                
                ax.scatter( np.array(dataframes[i]['x'].tail(1)), np.array(dataframes[i]['y'].tail(1)), np.array(dataframes[i]['z'].tail(1)),
                            color='white', s=0.1*bodies[i].m**2 )
                
                # plot_with_gradient(ax, np.array(dataframes[i]['x'].tail(500)), np.array(dataframes[i]['y'].tail(500)), np.array(dataframes[i]['z'].tail(500)), 0, 50)
                
                if show_forces == True:
                    r_start = np.array([ np.array(dataframes[i]['x'].tail(1)), np.array(dataframes[i]['y'].tail(1)), np.array(dataframes[i]['z'].tail(1)) ])

                    r_end_g =  np.array([ np.array(dataframes[i]['Fgx'].tail(1)), np.array(dataframes[i]['Fgy'].tail(1)), np.array(dataframes[i]['Fgz'].tail(1)) ])
                    ax.quiver(r_start[0], r_start[1], r_start[2],    r_end_g[0], r_end_g[1], r_end_g[2],     color='red')

                    r_end_d =  np.array([ np.array(dataframes[i]['Fdx'].tail(1)), np.array(dataframes[i]['Fdy'].tail(1)), np.array(dataframes[i]['Fdz'].tail(1)) ])
                    ax.quiver(r_start[0], r_start[1], r_start[2],    r_end_d[0], r_end_d[1], r_end_d[2],     color='blue')

                    r_end_t =  np.array([np.array(dataframes[i]['Fdx'].tail(1)) + np.array(dataframes[i]['Fgx'].tail(1)), 
                                         np.array(dataframes[i]['Fdy'].tail(1)) + np.array(dataframes[i]['Fgy'].tail(1)),
                                         np.array(dataframes[i]['Fdz'].tail(1)) + np.array(dataframes[i]['Fgz'].tail(1))])
                    ax.quiver(r_start[0], r_start[1], r_start[2],    r_end_t[0], r_end_t[1], r_end_t[2],     color='green')
        

        ##````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````##

        if not three_dimensional_plot:
            pass
        else:
            ax.set_axis_off()

        ax.set_facecolor("black")
        ax.set_aspect('equal')

        plt.pause(0.001)
        t += timestep
        
        '''
        if t > 200:
            break
        '''

        if run_check(bodies, 50):
            break