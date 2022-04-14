# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 19:26:24 2022

@author: aorti
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
# Initialization
tstart = 0
tstop = 60
increment = 0.1
# Initial condition
x_init = [0,0]
t = np.arange(tstart,tstop+1,increment)
# Function that returns dx/dt
def mydiff(x, t):
    c = 4 # Damping constant# N s/m
    k = 2 # Stiffness of the spring# N/m
    m = 20 # Mass# kg
    F = 5 # force amplitude constant (N)
    dx1dt = x[1]
    dx2dt = (F - c*x[1] - k*x[0])/m
    dxdt = [dx1dt, dx2dt]
    return dxdt
# Solve ODE
x = odeint(mydiff, x_init, t)
x1 = x[:,0]
x2 = x[:,1]
# Plot the Results
plt.plot(t,x1)
plt.plot(t,x2)
plt.title('Simulation of Mass-Spring-Damper System')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend(["x1", "x2"])
plt.grid()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
# Parameters defining the system
c = 5 # Damping constant
k = 2 # Stiffness of the spring
m = 20 # Mass
F = 5 # Force
Ft = np.ones(610)*F
# Simulation Parameters
tstart = 0
tstop = 60
increment = 0.1
t = np.arange(tstart,tstop+1,increment)
Ft = F#np.sin(t*0.5)*np.sin(t*0.2)*np.sin(t*0.7)*F
Ft = np.ones(610)*F

# System matrices
A = [[0, 1], [-k/m, -c/m]]
B = [[0], [1/m]]
C = [[1, 0]]
sys = sig.StateSpace(A, B, C, 0)
# Step response for the system
t, y, x = sig.lsim(sys, Ft, t)
x1 = x[:,0]
x2 = x[:,1]
plt.plot(t,Ft,t, x1, t, x2)
#plt.plot(t, y)
plt.title('Simulation of Mass-Spring-Damper System')
plt.xlabel('t')
plt.legend(["Ft","x1", "x2"])
plt.ylabel('x(t)')
plt.grid()
plt.show()



# Simulation of Mass-Spring-Damper System
import numpy as np
import matplotlib.pyplot as plt
# Model Parameters
c = 4 # Damping constant
k = 2 # Stiffness of the spring
m = 20 # Mass
F = 5 # Force
# Simulation Parameters
Ts = 0.1
Tstart = 0
Tstop = 60
N = int((Tstop-Tstart)/Ts) # Simulation length
x1 = np.zeros(N+2)
x2 = np.zeros(N+2)
x1[0] = 0 # Initial Position
x2[0] = 0 # Initial Speed
a11 = 1
a12 = Ts
a21 = -(Ts*k)/m
a22 = 1 - (Ts*c)/m
b1 = 0
b2 = Ts/m
# Simulation
for k in range(N+1):
    x1[k+1] = a11 * x1[k] + a12 * x2[k] + b1 * F
    x2[k+1] = a21 * x1[k] + a22 * x2[k] + b2 * F
# Plot the Simulation Results
t = np.arange(Tstart,Tstop+2*Ts,Ts)
#plt.plot(t, x1, t, x2)
plt.plot(t,x1)
plt.plot(t,x2)
plt.title('Simulation of Mass-Spring-Damper System')
plt.xlabel('t [s]')
plt.ylabel('x(t)')
plt.grid()
plt.legend(["x1", "x2"])
plt.show()




# Simulation of Mass-Spring-Damper System
import numpy as np
import matplotlib.pyplot as plt
# Model Parameters
c = [40,5,40,10] # Damping constant HSC,LSC, LSR, HSR
c = [80,10,10,80] # Damping constant HSC,LSC, LSR, HSR
c = [10,10,10,10] # Damping constant HSC,LSC, LSR, HSR
#c = [1,1,40,25] # Damping constant HSC,LSC, LSR, HSR

k = 60 # Stiffness of the spring
m = 90 # Mass
F = 3000 # Force

# Simulation Parameters
Ts = 0.01
Tstart = 0
Tstop = 60
N = int((Tstop-Tstart)/Ts) # Simulation length
t = np.arange(Tstart,Tstop+2*Ts,Ts)
w1,w2,w3 = 0.1,0.1, 0.1
w1,w2,w3 = 0.4,0.3, 0.5
Ft = np.sin(t*w1) * np.sin(t*w2) * np.sin(t*w3)*F
Ft[Ft < 0] = 0

x1 = np.zeros(N+2)
x2 = np.zeros(N+2)
x1[0] = 0.15 # Initial Position
x2[0] = 0 # Initial Speed
a11 = 1
a12 = Ts
a21 = -(Ts*k)/m
a22=[]
for ci in c:
    a22.append(1 - (Ts*ci)/m)
    
    
HSLS = 2
b1 = 0
b2 = Ts/m
# Simulation
for k in range(N+1):
    if x1[k] < 0:
        x1[k],x2[k] = 0,0
    if x1[k] > 50:
        x1[k],x2[k] = 50,0
        
    if x2[k] > HSLS:#HSC
        
        x1[k+1] = a11 * x1[k] + a12 * x2[k] + b1 * Ft[k]
        x2[k+1] = a21 * x1[k] + a22[0] * x2[k] + b2 * Ft[k] 
    elif x2[k] >0:#LSC
        
        x1[k+1] = a11 * x1[k] + a12 * x2[k] + b1 * Ft[k]
        x2[k+1] = a21 * x1[k] + a22[1] * x2[k] + b2 * Ft[k] 
    elif x2[k] > -HSLS:#LSR
        
        x1[k+1] = a11 * x1[k] + a12 * x2[k] + b1 * Ft[k]
        x2[k+1] = a21 * x1[k] + a22[2] * x2[k] + b2 * Ft[k] 

    else:#HSR
        
        x1[k+1] = a11 * x1[k] + a12 * x2[k] + b1 * Ft[k] 
        x2[k+1] = a21 * x1[k] + a22[3] * x2[k] + b2 * Ft[k] 
# Plot the Simulation Results
plt.plot(t,x1)
plt.plot(t,x2)
plt.plot(t,Ft)
plt.title('Simulation of Mass-Spring-Damper System')
plt.xlabel('t[s]')
plt.ylabel('x(t)')
plt.grid()
plt.legend(["x1", "x2","Ft"])
plt.show()


plt.rcParams["figure.figsize"] = [15, 8]
plt.rcParams["figure.autolayout"] = True

speed = np.array([3, 1, 2, 0, 5])
acceleration = np.array([6, 5, 7, 1, 5])

ax1 = plt.subplot()
l1, = ax1.plot(t,x1, color='red')
ax2 = ax1.twinx()
l2, = ax2.plot(t,x2, color='orange')
ax3 = ax1.twinx()
l3, = ax3.plot(t,Ft, color='Blue')
ax3.set_ylim(0, max(Ft))


plt.legend([l1, l2, l3], ["Position", "Speed","Force"])

plt.show()