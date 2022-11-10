#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:26:58 2022

@author: nilmasocastro
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.patches as mpatches

#Random seed for random numbers
np.random.seed(206)


#%%

def initial_config(L, N, sigma):
    '''
    Initialices a configuration with no overlapping in an LxL box with N disks.
    Starts with a particle and then keeps adding particles taking the overlapping
    into account.
    Instead of checking PBC, it simply does not accept new particles that overlap 
    with the walls.
    
    INPUTS: L, N, sigma
    OUTPUTS: x_0, y_0 (initial conditions)
    '''
    
    x_0 = []
    y_0 = []
    i = 0 #the current disk we are trying to insert
    
    while i < N:
        x_try = np.random.uniform(-L/2, L/2)
        y_try = np.random.uniform(-L/2, L/2)
        
        if i == 0:
            x_0.append(x_try)
            y_0.append(y_try)
            i = 1
            continue #jumps directly to another iteration of the loop
        
        if ((np.abs(x_try)+(sigma/2))>L/2) or ((np.abs(y_try)+(sigma/2))>L/2):
            continue #jumps directly to another iteration of the loop and create another x_try and y_try
                     #since this last wasn't accepted
            
        #r is a vector of i elements of the distance between every particle with the i particle
        r = np.sqrt(np.square(np.subtract(x_try, x_0[:])) + np.square(np.subtract(y_try, y_0[:])))
            
        #We check if the minimum value of r is lower than sigma
        if min(r) < sigma:
            continue #jumps directly to another iteration of the loop and create another x_try and y_try
                     #since this last wasn't accepted
        else:
            x_0.append(x_try) #since there is no overlapping we accept the new i particle position
            y_0.append(y_try)
            i += 1 #let's go to the next particle

    x_0 = np.array(x_0) #transform the lists into numpy arrays
    y_0 = np.array(y_0)
    
    return x_0, y_0



def trial_acceptance(x, y, h_count, v_count, L, N, sigma, delta, t):
    '''
    Trial acceptance function that does these 2 steps N times, and obtain an x and y 
    position vectors whose columns are determined times (Monte Carlo Steps)
    We try to move a particle and accept or reject depending if there is overlapping
    
    INPUTS: x(N,t_steps), y(N,t_steps), h_count(N,t_steps), v_count(N,t_steps), 
            L, N, sigma, delta, t(specific Monte Carlo step, integer)
    OUTPUTS: x(changed), y(changed)
    '''
    
    #Copy of the input positions, to not overwrite with the new positions
    x_old = np.copy(x[:,t-1])
    y_old = np.copy(y[:,t-1])
    h_count_old = np.copy(h_count[:,t-1])
    v_count_old = np.copy(v_count[:,t-1])

    #We try N times the 2 step algorithm (try number is "i")
    for i in range(0,N):
        
        x_try = np.copy(x_old)
        y_try = np.copy(y_old)
        h_count_try = np.copy(h_count_old)
        v_count_try = np.copy(v_count_old)
        
        #Random choice of a particle of all the N particles 
        i_particle = random.randint(0,N-1)
        
        #New possible position of a random particle "i" on a certain time "t"
        x_try[i_particle] = x_old[i_particle] + delta * (np.random.uniform(0,1) - 0.5)
        y_try[i_particle] = y_old[i_particle] + delta * (np.random.uniform(0,1) - 0.5)
        
        #Find the values of h_count_try and v_count_try for the possible new movement (because of PBC)
        h = int(x_try[i_particle]//(L/2))
        v = int(y_try[i_particle]//(L/2))
        
        if h < 0:
            h_count_try[i_particle] = h + 1
        else:
            h_count_try[i_particle] = h
        if v < 0:
            v_count_try[i_particle] = v + 1
        else:
            v_count_try[i_particle] = v
        
        #Accept the new position?
        #Distance between centers: r
        
        #r is a vector of N elements of the distance between every particle and i_particle
        r = np.sqrt(np.square(np.subtract(x_try[i_particle]-L*h_count_try[i_particle], x_try-L*h_count_try)) + np.square(np.subtract(y_try[i_particle]-L*v_count_try[i_particle], y_try-L*v_count_try)))
        r = np.delete(r,i_particle) #remove the value corresponding to the same i_particle
        
        #We check if the minimum value of r is higher than sigma
        if min(r) >= sigma:
            #Since we accept the new position we change it in the x and y vectors
            x_old[i_particle] = x_try[i_particle]
            y_old[i_particle] = y_try[i_particle]
            #Change the h_count and v_count for the new movement
            h_count_old[i_particle] = h_count_try[i_particle]
            v_count_old[i_particle] = v_count_try[i_particle]
        
        else:
            continue

    #Return the new configuration, after N trys
    x[:,t] = x_old
    y[:,t] = y_old
    h_count[:,t] = h_count_old
    v_count[:,t] = v_count_old
            
    return x, y, h_count, v_count


def MSD(x, y, N, t):
    '''
    Computes the Mean Square Displacement from LxL box system for a certain time t.
    We do not need the h and v values since we consider an open box.
    
    INPUTS: x(N,t_steps), y(N,t_steps), N, t
    OUTPUTS: MSD_value
    '''
    
    r_i = np.zeros(N)
    r_i = np.square(x[:,t] - x[:,0] + y[:,t] - y[:,0])
    
    MSD_value = sum(r_i)/N
    
    return MSD_value
    

#%% PART 1

#Number of hard disks
N = 1000

#Diameter of the disks
sigma = 1.0

#Area fraction
phi = 0.05
L = sigma * np.sqrt((np.pi * N)/(4 * phi))

#Delta/sigma values
delta_values = sigma * np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3])

#Number of Monte Carlo steps (time variable)
t_steps = 1000

#MSD array. Every column is a different MC step and every row is a different delta value 
MSD_values = np.zeros((len(delta_values),t_steps))

for d in range(0,len(delta_values)):
    
    print('Delta value: ', delta_values[d])    

    x = np.zeros((N,t_steps))
    y = np.zeros((N,t_steps))
    
    x_0, y_0 = initial_config(L, N, sigma)
    
    for k in range(0,t_steps):
        x[:,k] = x_0
        y[:,k] = y_0
    
    h_count = np.zeros((N,t_steps), dtype=int)
    v_count = np.zeros((N,t_steps), dtype=int)
    
    
    for t in range(1,t_steps):
        x, y, h_count, v_count = trial_acceptance(x, y, h_count, v_count, L, N, sigma, delta_values[d], t)
        print('Monte Carlo step: ', t)
        
    
    for t in range(0,t_steps):
        MSD_values[d,t] = MSD(x, y, N, t)


D_values = (MSD_values[:,t_steps-1])/(4 * t_steps)



#%% TO CHECK SOME OVERLAPPING PROBLEMS
'''
sigma = 1.0
count = 0

for i_particle in range(0,N):
    t=1

    r = np.sqrt(np.square(np.subtract(x[i_particle,t]-L*h_count[i_particle,t], x[:,t]-L*h_count[:,t])) \
                + np.square(np.subtract(y[i_particle,t]-L*v_count[i_particle,t], y[:,t]-L*v_count[:,t])))
    #r = np.sqrt(np.square(np.subtract(x_0[i_particle], x_0[:])) \
     #           + np.square(np.subtract(y_0[i_particle], y_0[:])))
    r = np.delete(r,i_particle) #remove the value corresponding to the same i_particle
    
    if min(r) < sigma:
        count += 1
        print(min(r))

print(count)
'''


#%% Plots of PART 1

for i in range(0,len(delta_values)):
    plt.plot(MSD_values[i,:], '.-', markersize=3.0, label='$\delta$ = ' + str(delta_values[i]) + '$\sigma$')

plt.legend()
plt.xlabel('MC step')
plt.ylabel('MSD ($\sigma^{2}$)')    
#plt.xlim(0,t_steps)
#plt.ylim(0,200)
plt.xscale('log')
plt.yscale('log')

plt.show()

#%%

plt.plot(delta_values, D_values, '.-', markersize=5.0)
plt.xlabel('$\delta$ ($\sigma$)')
plt.ylabel('D ($\sigma^{2}$ / MC step)') 
plt.xscale('log')
plt.yscale('log')

plt.show()


##############################################################################
#%% PART 2

def triangular_lattice(L, N, sigma, sep):
    '''
    Initializes configuration with N particles and area fraction phi distributed in
    a triangular lattice. L is determined by the phi (computed outside the function).

    Parameters
    ----------
    L : float
        Length of one side of the LxL box.
    N : integer
        Number of disks in the system.
    sigma : float
        Diameter of the disks.
    sep : float
        Extra separation between particles.

    Returns
    -------
    x_0 : float array (N)
        x initial position of the N disks.
    y_0 : float array (N)
        y initial position of the N disks.

    '''
    
    a = sigma + sep #distance between disks centers
    
    r_c = int(np.sqrt(N)) + 1 #number of rows and columns needed
    
    x_0 = []
    y_0 = []
    
    row_disks = []
    
    for i in range(0,r_c):
        row_disks.append(i*a)
    
    for j in range(0,r_c-1):
        
        if (j % 2) == 0: #if the number is even
            x_0.extend(row_disks)
            for k in range(0,r_c):
                y_0.append((j//2)*(np.sqrt(3)*a))
        
        else:
            x_0.extend([x+(a/2) for x in row_disks])
            for k in range(0,r_c):
                y_0.append((j*np.sqrt(3)*a)/2)
        
    disks_left = N - (r_c)*(r_c-1)
    for i in range(0,disks_left):
        if ((r_c-1) % 2) == 0: #if the number is even
            x_0.append(i*a)
            y_0.append(((r_c-1)//2)*(np.sqrt(3)*a))
        else:
            x_0.append(i*a+(a/2))
            y_0.append(((r_c-1)*np.sqrt(3)*a)/2)
        
    x_0 = np.array(x_0) #transform the lists into numpy arrays
    y_0 = np.array(y_0)
    
    x_0 = np.subtract(x_0,max(x_0)/2) #positions the coordinates in the center
    y_0 = np.subtract(y_0,max(y_0)/2) #of the box
    
    return x_0, y_0


#%%

#Number of hard disks
N = 1000

#Diameter of the disks
sigma = 1.0

delta = 0.3

sep = 0.2 #a bit of separation

t_steps = 7000

#Area fraction
phi_values = np.array([0.05, 0.2, 0.5]) #since we are changing the value of area fraction

x_all = np.zeros((len(phi_values),N,t_steps))
y_all = np.zeros((len(phi_values),N,t_steps))
h_count_all = np.zeros((len(phi_values),N,t_steps), dtype=int)
v_count_all = np.zeros((len(phi_values),N,t_steps), dtype=int)
MSD_values_all = np.zeros((len(phi_values),t_steps))

for p in range(0,len(phi_values)):

    print('Phi value: ', phi_values[p])
    
    L = sigma * np.sqrt((np.pi * N)/(4 * phi_values[p]))
    
    x = np.zeros((N,t_steps))
    y = np.zeros((N,t_steps))
    
    x_0, y_0 = triangular_lattice(L, N, sigma, sep)
    
    for k in range(0,t_steps):
            x[:,k] = x_0
            y[:,k] = y_0
    
    h_count = np.zeros((N,t_steps), dtype=int)
    v_count = np.zeros((N,t_steps), dtype=int)
        
    for t in range(1,t_steps):
        x, y, h_count, v_count = trial_acceptance(x, y, h_count, v_count, L, N, sigma, delta, t)
        print('Monte Carlo step: ', t)
    
    MSD_values = np.zeros(t_steps)
    
    for t in range(0,t_steps):
        MSD_values[t] = MSD(x, y, N, t)


    x_all[p,:,:] = x
    y_all[p,:,:] = y
    h_count_all[p,:,:] = h_count
    v_count_all[p,:,:] = v_count
    MSD_values_all[p,:] = MSD_values



#%% Plots for PART 2
for p in range(0,len(phi_values)):
    plt.plot(MSD_values_all[p,:], '-', label='$\phi$ = ' + str(phi_values[p]))
    
plt.legend()
plt.xlabel('MC step')
plt.ylabel('MSD ($\sigma^{2}$)')
plt.xscale('log')
plt.yscale('log')

plt.show()


#%%
#Diameter of the disks
sigma = 1.0

phi_values = np.array([0.05, 0.2, 0.5])

size_disk = np.array([0.6, 5.0, 15.0]) #to see the particles just touching

p = 2

time = 0

L = sigma * np.sqrt((np.pi * N)/(4 * phi_values[p]))
    
plt.scatter(x_all[p,:,time]-np.dot(L,h_count_all[p,:,time]), y_all[p,:,time]-np.dot(L,v_count_all[p,:,time]), marker='o', s=size_disk[p], edgecolors='black', facecolors='black')
plt.xlabel('x ($\sigma$)')
plt.ylabel('y ($\sigma$)')
plt.axis('square')
plt.xlim(-L/2-5, L/2+5)
plt.ylim(-L/2-5, L/2+5)
    
rect=mpatches.Rectangle((-L/2,-L/2),L,L, 
                        fill = False,
                        color = "purple",
                        linewidth = 1)
        	
plt.gca().add_patch(rect)
    
plt.show()


#%% CHECK IF THERE IS NO OVERLAPPING
'''
phi_values = np.array([0.05, 0.2, 0.5])

p = 0

L = sigma * np.sqrt((np.pi * N)/(4 * phi_values[p]))

for i_particle in range(0,N):
    for t in range(0,10):

        r = np.sqrt(np.square(np.subtract(x_all[p,i_particle,t]-L*h_count_all[p,i_particle,t], x_all[p,:,t]-L*h_count_all[p,:,t])) \
                    + np.square(np.subtract(y_all[p,i_particle,t]-L*v_count_all[p,i_particle,t], y_all[p,:,t]-L*v_count_all[p,:,t])))
        r = np.delete(r,i_particle) #remove the value corresponding to the same i_particle
        #print(np.shape(r))
        if min(r) < sigma:
            print(min(r))
'''



##############################################################################
#%% PART 3

#Now in a closed box (without PBC) and its sides are of different length (L_x x L_y)

def initial_config_box(Lx, Ly, N, sigma):
    '''
    Initialices a configuration with no overlapping in an LxL box with N disks.
    Starts with a particle and then keeps adding particles taking the overlapping
    into account.
    Instead of checking PBC, it simply does not accept new particles that overlap 
    with the walls.

    Parameters
    ----------
    Lx : float
        Horizontal lenght of the box.
    Ly : float
        Vertical lenght of the box.
    N : integer
        Number of disks.
    sigma : float
        Diameter of the disks.

    Returns
    -------
    x_0 : float array (N)
        x initial position of the N disks.
    y_0 : float array (N)
        y initial position of the N disks.

    '''
    
    x_0 = []
    y_0 = []
    i = 0 #the current disk we are trying to insert
    
    while i < N:
        x_try = np.random.uniform(-Lx/2, Lx/2)
        y_try = np.random.uniform(-Ly/2, Ly/2)
        
        if i == 0:
            x_0.append(x_try)
            y_0.append(y_try)
            i = 1
            continue #jumps directly to another iteration of the loop
        
        if ((np.abs(x_try)+(sigma/2))>Lx/2) or ((np.abs(y_try)+(sigma/2))>Ly/2):
            continue #jumps directly to another iteration of the loop and create another x_try and y_try
                     #since this last wasn't accepted
            
        #r is a vector of i elements of the distance between every particle with the i particle
        r = np.sqrt(np.square(np.subtract(x_try, x_0[:])) + np.square(np.subtract(y_try, y_0[:])))
            
        #We check if the minimum value of r is lower than sigma
        if min(r) < sigma:
            continue #jumps directly to another iteration of the loop and create another x_try and y_try
                     #since this last wasn't accepted
        else:
            x_0.append(x_try) #since there is no overlapping we accept the new i particle position
            y_0.append(y_try)
            i += 1 #let's go to the next particle

    x_0 = np.array(x_0) #transform the lists into numpy arrays
    y_0 = np.array(y_0)
    
    
    return x_0, y_0



def trial_acceptance_box(x, y, Lx, Ly, N, sigma, delta, t):
    '''
    Trial acceptance function that does these 2 steps N times, and obtain an x and y 
    position vectors whose columns are determined times (Monte Carlo Steps)
    We try to move a particle and accept or reject depending if there is overlapping,
    also with the walls
    
    INPUTS: x(N,t_steps), y(N,t_steps), Lx, Ly, N, sigma, delta, t(specific Monte Carlo step, integer)
    OUTPUTS: x(changed), y(changed)
    '''
    
    
    #Copy of the input positions, to not overwrite with the new positions
    x_old = np.copy(x[:,t-1])
    y_old = np.copy(y[:,t-1])
    
    #We try N times the 2 step algorithm (try number is "i")
    for i in range(0,N):
        
        x_try = np.copy(x_old)
        y_try = np.copy(y_old)
        
        #Random choice of a particle of all the N particles 
        i_particle = random.randint(0,N-1)
        
        #New possible position of a random particle "i" on a certain time "t"
        x_try[i_particle] = x_old[i_particle] + delta * (np.random.uniform(0,1) - 0.5)
        y_try[i_particle] = y_old[i_particle] + delta * (np.random.uniform(0,1) - 0.5)
        
        
        #Accept the new position?
        #Distance between centers: r
        
        #r is a vector of N elements of the distance between every particle and i_particle
        r = np.sqrt(np.square(np.subtract(x_try[i_particle], x_try)) + np.square(np.subtract(y_try[i_particle], y_try)))
        r = np.delete(r,i_particle) #remove the value corresponding to the same i_particle
        
        #We check if the minimum value of r is smaller than sigma (we reject)
        if min(r) < sigma:
            continue
        #Also check if the new position overlaps with the walls
        elif ((np.abs(x_try[i_particle])+(sigma/2))>Lx/2) or ((np.abs(y_try[i_particle])+(sigma/2))>Ly/2):
            continue
        
        else:
            #Since we accept the new position we change it in the x and y vectors
            x_old[i_particle] = x_try[i_particle]
            y_old[i_particle] = y_try[i_particle]

    #Return the new configuration, after N trys
    x[:,t] = x_old
    y[:,t] = y_old
            
    return x, y


#%% MC simulation
x = np.zeros((N,t_steps))
y = np.zeros((N,t_steps))

#Number of hard disks
N = 1000

#Diameter of the disks
sigma = 1.0

delta = 0.3

t_steps = 1000

#Area fraction
phi = 0.05
Ly = sigma * np.sqrt((np.pi * N)/(4 * phi))
Lx = 2*Ly

x_0, y_0 = initial_config_box(Lx, Ly, N, sigma)

for k in range(0,t_steps):
    x[:,k] = x_0
    y[:,k] = y_0

for t in range(1,t_steps):
    x, y = trial_acceptance_box(x, y, Lx, Ly, N, sigma, delta, t)
    print(t)



#%% To plot the final configuration of a closed box Lx x Ly

plt.figure(figsize=(7,4.1))
plt.scatter(x[:,0], y[:,0], s=sigma, marker='o', edgecolors='black', facecolors='none')
plt.xlabel('x ($\sigma$)')
plt.ylabel('y ($\sigma$)')
plt.axis('square')
plt.xlim(-Lx/2-5, Lx/2+5)
plt.ylim(-Ly/2-5, Ly/2+5)
    
rect=mpatches.Rectangle((-Lx/2,-Ly/2),Lx,Ly, 
                        fill = False,
                        color = "purple",
                        linewidth = 1)
        	
plt.gca().add_patch(rect)
    
plt.show()




#%% PART 4 AND 5

def trial_acceptance_gravity(x, y, Lx, Ly, N, sigma, delta, t, g):
    '''
    Trial acceptance function that does these 2 steps N times, and obtain an x and y 
    position vectors whose columns are determined times (Monte Carlo Steps)
    We try to move a particle and accept or reject depending if there is overlapping,
    also with the walls. Particles are subjected to gravity.
    
    INPUTS: x(N,t_steps), y(N,t_steps), Lx, Ly, N, sigma, delta,
            t(specific Monte Carlo step, integer), g(float)
    OUTPUTS: x(changed), y(changed)
    '''
    
    
    #Copy of the input positions, to not overwrite with the new positions
    x_old = np.copy(x[:,t-1])
    y_old = np.copy(y[:,t-1])
    
    #We try N times the 2 step algorithm (try number is "i")
    for i in range(0,N):
        
        x_try = np.copy(x_old)
        y_try = np.copy(y_old)
        
        #Random choice of a particle of all the N particles 
        i_particle = random.randint(0,N-1)
        
        #New possible position of a random particle "i" on a certain time "t"
        x_try[i_particle] = x_old[i_particle] + delta * (np.random.uniform(0,1) - 0.5)
        y_try[i_particle] = y_old[i_particle] + delta * (np.random.uniform(0,1) - 0.5)
        
        
        #Accept the new position?
        #Distance between centers: r
        
        #r is a vector of N elements of the distance between every particle and i_particle
        r = np.sqrt(np.square(np.subtract(x_try[i_particle], x_try)) + np.square(np.subtract(y_try[i_particle], y_try)))
        r = np.delete(r,i_particle) #remove the value corresponding to the same i_particle
        
        #We check if the minimum value of r is smaller than sigma (we reject)
        if min(r) < sigma:
            continue
        #Also check if the new position overlaps with the walls
        elif ((np.abs(x_try[i_particle])+(sigma/2))>Lx/2) or ((np.abs(y_try[i_particle])+(sigma/2))>Ly/2):
            continue
        
        else:
            
            delta_y = y_try[i_particle] - y_old[i_particle]
            beta = 1.0
            m = 1.0
            prob = np.exp(-beta*m*g*delta_y)
            
            if min(1,prob)!=prob:
            
                #Since we accept the new position we change it in the x and y vectors
                x_old[i_particle] = x_try[i_particle]
                y_old[i_particle] = y_try[i_particle]

    #Return the new configuration, after N trys
    x[:,t] = x_old
    y[:,t] = y_old
            
    return x, y

#%%

gravity_values = np.array([0, 0.01, 0.1, 1.0, 10.0])

#Number of hard disks
N = 1000

#Diameter of the disks
sigma = 1.0

delta = 0.3

t_steps = 7000

#Area fraction
phi = 0.05
Lx = sigma * np.sqrt((np.pi * N)/(4 * phi))
Ly = 10*Lx


x_all = np.zeros((len(gravity_values),N,t_steps))
y_all = np.zeros((len(gravity_values),N,t_steps))

for g in range(0,len(gravity_values)):

    print('Gravity value: ', gravity_values[g])
    
    x = np.zeros((N,t_steps))
    y = np.zeros((N,t_steps))
    
    x_0, y_0 = initial_config_box(Lx, Ly, N, sigma)
    
    for k in range(0,t_steps):
            x[:,k] = x_0
            y[:,k] = y_0
        
    for t in range(1,t_steps):
        x, y = trial_acceptance_gravity(x, y, Lx, Ly, N, sigma, delta, t, gravity_values[g])
        if (t%1000 == 0):
            print('Monte Carlo step: ', t)

    x_all[g,:,:] = x
    y_all[g,:,:] = y



#%% Plots for PART 5

time = 6999

plt.figure()
plt.subplot(1,5,1)
plt.scatter(x_all[0,:,0], y_all[0,:,0], s=sigma, marker='o', edgecolors='black', facecolors='none')
#plt.xlabel('x ($\sigma$)')
plt.ylabel('y ($\sigma$)')
plt.title('g=0.0')
plt.axis('square')
plt.xlim(-Lx/2-5, Lx/2+5)
plt.ylim(-Ly/2-5, Ly/2+5)
    
rect=mpatches.Rectangle((-Lx/2,-Ly/2),Lx,Ly, 
                        fill = False,
                        color = "purple",
                        linewidth = 1)
plt.gca().add_patch(rect)
    
plt.subplot(1,5,2)
plt.scatter(x_all[1,:,time], y_all[1,:,time], s=sigma, marker='o', edgecolors='black', facecolors='none')
#plt.xlabel('x ($\sigma$)')
#plt.ylabel('y ($\sigma$)')
plt.title('g=0.01')
plt.axis('square')
plt.xlim(-Lx/2-5, Lx/2+5)
plt.ylim(-Ly/2-5, Ly/2+5)
    
rect=mpatches.Rectangle((-Lx/2,-Ly/2),Lx,Ly, 
                        fill = False,
                        color = "purple",
                        linewidth = 1)
plt.gca().add_patch(rect)

plt.subplot(1,5,3)
plt.scatter(x_all[2,:,time], y_all[2,:,time], s=sigma, marker='o', edgecolors='black', facecolors='none')
plt.xlabel('x ($\sigma$)')
#plt.ylabel('y ($\sigma$)')
plt.title('g=0.1')
plt.axis('square')
plt.xlim(-Lx/2-5, Lx/2+5)
plt.ylim(-Ly/2-5, Ly/2+5)
    
rect=mpatches.Rectangle((-Lx/2,-Ly/2),Lx,Ly, 
                        fill = False,
                        color = "purple",
                        linewidth = 1)
plt.gca().add_patch(rect)

plt.subplot(1,5,4)
plt.scatter(x_all[3,:,time], y_all[3,:,time], s=sigma, marker='o', edgecolors='black', facecolors='none')
#plt.xlabel('x ($\sigma$)')
#plt.ylabel('y ($\sigma$)')
plt.title('g=1.0')
plt.axis('square')
plt.xlim(-Lx/2-5, Lx/2+5)
plt.ylim(-Ly/2-5, Ly/2+5)
    
rect=mpatches.Rectangle((-Lx/2,-Ly/2),Lx,Ly, 
                        fill = False,
                        color = "purple",
                        linewidth = 1)
plt.gca().add_patch(rect)

plt.subplot(1,5,5)
plt.scatter(x_all[4,:,time], y_all[4,:,time], s=sigma, marker='o', edgecolors='black', facecolors='none')
#plt.xlabel('x ($\sigma$)')
#plt.ylabel('y ($\sigma$)')
plt.title('g=10.0')
plt.axis('square')
plt.xlim(-Lx/2-5, Lx/2+5)
plt.ylim(-Ly/2-5, Ly/2+5)
    
rect=mpatches.Rectangle((-Lx/2,-Ly/2),Lx,Ly, 
                        fill = False,
                        color = "purple",
                        linewidth = 1)
plt.gca().add_patch(rect)

plt.show()
