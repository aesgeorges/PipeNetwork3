# Alexandre Erich Sébastien Georges
# Stony Brook University, Undergraduate Student in Civil Engineering - minor, Computer Science
# Class of 2020

# Code to use convergence method to find velocities in a network of pipes
# (here 3 pipes and 3 reservoirs)

import math
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec

# Value of v1 is the original guess
v1 = 0.5

# Values given for the problem setup
f1 = 0.015
f2 = 0.02
f3 = 0.02
D1 = 0.1
D2 = 0.08
D3 = 0.08
l1 = 200
l2 = 200
l3 = 400
z1 = 60
z2 = 20
z3 = 0

# vectors for plotting convergence
varray1 = []
varray2 = []
varray3 = []
convarray = []

# Initializing variables needed for convergence method
i = 0
conv = 0
new_v1 = 0

# Initializing the Convergence Table
table = PrettyTable()
table.field_names = ['it#', 'v1', 'v2', 'v3', 'new v1', 'Convergence %']


# Finding v2 using Bernouilli equation
def calculate_v2(v1):
    v2 = math.sqrt((D2 / (f2 * l2)) * (20 * (z1 - z2) - ((f1 * l1) / D1) * math.pow(v1, 2)))
    return v2


# Finding v3 using Bernouilli equation
def calculate_v3(v1):
    v3 = math.sqrt((D3 / (f3 * l3)) * (20 * (z1 - z3) - ((f1 * l1) / D1) * math.pow(v1, 2)))
    return v3


# Convergence table function
def convergence_table(v1):
    global i, table, new_v1, conv, varray1, varray2, varray3, convarray
    v2 = calculate_v2(v1)
    v3 = calculate_v3(v1)
    # Formula for new_v1 might need to be changed depending on pipe setup and assumption of direction of flow
    new_v1 = (math.pow(D2, 2)/math.pow(D1, 2)) * v2 + (math.pow(D3, 2)/math.pow(D1, 2)) * v3
    conv = (v1 - new_v1) / v1
    # adding data to numpy array
    varray1.append(v1)
    varray2.append(v2)
    varray3.append(v3)
    convarray.append(conv)
    # adding data to PrettyTable
    table.add_row([i, v1, v2, v3, new_v1, conv*100])
    i += 1


# Calling Convergence table function and putting it on a loop until the convergence percentage is low enough
min_conv = 0.0005
convergence_table(v1)
while abs(conv) > min_conv:
    convergence_table(new_v1)

# matplotlib scatter plot

fig = plt.figure()
varray1 = np.array(varray1)
varray2 = np.array(varray2)
varray3 = np.array(varray3)
x = np.linspace(0, len(varray1) - 1, len(varray1))

plt.subplot(411)
plt.ylabel('v1')
plt.xlabel('iteration #')
plt.plot(x, varray1, color='b')

plt.subplot(412)
plt.ylabel('v2')
plt.xlabel('iteration #')
plt.plot(x, varray2, color='r')

plt.subplot(413)
plt.ylabel('v3')
plt.xlabel('iteration #')
plt.plot(x, varray3, color='g')

plt.subplot(414)
plt.ylabel('Convergence %')
plt.xlabel('iteration #')
plt.plot(x, convarray, color='k')


# Printing final Table
print(table)
plt.show()
