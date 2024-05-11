"""
Created on Sat Apr 27 17:44:31 2024

@author: jrjol
"""
'''
Scipy Interp2D has unfortunately been deprecated. 

Alternatives have issues: 
    - Scipy.interpolate.CloughTocher2DInterpolate returns 'nan' for any out of 
    bounds extrapolation.
    - Scipy.interpolate.RegularGridInterpolator breaks the code entirely.
    - Scipy.interpolate.NearestNDInterpolator returns the best known boundary 
    values to the extrapolation point.
    -Scipy.interpolate.RecBivarateSpline retunrs the best known boundary values 
    to the extrapolation point.

Hence I have made my own function 'interp2d' which interpolates, and attempts
linear extrapolation, for out of bounds predictions. 

'''



import math as m
import numpy as np
import pygad
from scipy.interpolate import interp2d
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import CloughTocher2DInterpolator
import matplotlib.pyplot as plt


aerofoil_name = 'naca 4415'


def Cp_Calc(delta_Q, OMEGA, a, U, swept_radius):
    '''
    Correct

    Parameters
    ----------
    delta_Q : List
        Takes in each delta_Q for each element along the blade.
    OMEGA : Float
        Angular velocity. f(TSR, U, R)
    a : Float
        DESCRIPTION.
    U : Float
        Incoming wind speed velocity (m/s).
    swept_radius : Float
        The swept radius of the wind turbine design.

    Returns
    -------
    Cp : Float
        Coefficient of Performance.

    '''
    A = sum(delta_Q)
    print('delta Qs : ',delta_Q)
    print(' Omegas: ', OMEGA)
    Power = A*OMEGA
    Cp = Power/(1/2*1.204*(m.pi*swept_radius**2)*U**3)
    print('cp: ', Cp)
    return Cp

#%%

'''

notes:
    CL CD not found returns error. try extending number of iterations first. if not, return 0 0. and leave to extrapolator. 
    test the occurance of this issue in the region of A<0.
    

'''



from io import *
import os
import re

def call_Xfoil(alpha, Re, airfoil_name):
    
    #open the file for the xfoil commands
    out=open('xfoilinput.txt', 'w')
    
    # Write xfoil commands to the file (as they would 
    # be typed into xfoil)
    out.write(
    f"""
    {airfoil_name}
    panel
    OPER
    iter 200
    visc
    {Re}
    A {alpha}
    !
    
    quit
    """
    )
    
    #Close the file to make sure the file is written 
    #to disk properly
    out.close() 
    
    #call the system command to run xfoil and redirect 
    #our input file into it
    os.system('xfoil < xfoilinput.txt > xfoiloutput.txt')
    
    #Using the python popen command, the output from this can 
    #be redirected from the standard output using the read() 
    #method.
    
    # Read the file and get the last 10 lines
    text = open("xfoiloutput.txt").readlines()[-10:]
    
    # Regular expression patterns
    cl_pattern = r'CL =\s*([0-9.]+)'  # Matches CL value
    cd_pattern = r'CD =\s*([0-9.]+)'  # Matches CD value
    
    # Search for CL value
    matches = re.search(cl_pattern, text[5])
    if matches:
        cl = float(matches.group(1))
        print("Lift coefficient =", cl)
    else:
        cl = 0
        print("CL not found")
    
    # Search for CD value
    matches = re.search(cd_pattern, text[6])
    if matches:
        cd = float(matches.group(1))
        print("Drag coefficient =", cd)
    else:
        cd = 0
        print("CD not found")
    
    return cl, cd

Cl, Cd = call_Xfoil(-5, 2300, airfoil_name='naca 4415')
    


#%%

#def GEN_CL_CD():
hashmap_Cd = {}
hashmap_Cl = {}
#Reynolds = np.linspace(0, 110000, int((110000-2500)/5000))
Reynolds = np.array([2300, 5000, 7500, 10000,15000, 20000,30000, 40000]) #, 80000, 160000, 320000])

Attacks = np.linspace(-10,20)

X, Y = np.meshgrid(Reynolds,Attacks)
CLs = np.empty((0, len(Attacks)))
CDs = np.empty((0, len(Attacks)))
for Re in Reynolds: # y-axis
    piecewise = 'False'
    CL=[]
    CD=[]
    for i, attack in enumerate(Attacks): # x-axis
        print('run: ', i)
        if Re<2300:
            
            #Cl, Cd = call_Xfoil(attack, Re, airfoil_name='naca 4415')
            Cl = Cd = 0
            #print(Cl,Cd)
        else:
            Cl, Cd = call_Xfoil(attack, Re, airfoil_name=aerofoil_name)
        CL.append(Cl)
        CD.append(Cd)
        #print('Currently, Re and Alpha: ', Re, attack)
        if piecewise== 'True':
            CL[i-1] =(CL[i]-CL[i-2])/2+CL[i-2]
            CD[i-1] = (CD[i]-CD[i-2])/2+CD[i-2]
            piecewise= 'False'
        if Cl==0:
            piecewise ='True'
    CL = np.array(CL)
    CD = np.array(CD)
    # CL = np.expand_dims(CL, axis=1)  # Add a new axis to make it a column vector
    # CD = np.expand_dims(CD, axis=1) 
    
    CLs = np.vstack((CLs, CL))
    CDs = np.vstack((CDs, CD))
    
    #hashmap_Cd[int(Re)] = CDs
    #hashmap_Cl[int(Re)] = CLs

# x = Reynolds.repeat(len(Attacks))  # Repeat each Reynolds value for all Attacks
# y = np.tile(Attacks, len(Reynolds))  # Tile Attacks for each Reynolds value
# z = CLs.flatten()

# f = CloughTocher2DInterpolator((x, y), z)

# # # Interpolate/extrapolate at specific points
# x_point = 9000
# y_point = 21
# cl_value = f(x_point, y_point)


### RectBivarate Spline / Regular grid Interpolator ####
# Sample data points
x = Reynolds
y = Attacks
# # #z = np.array([[1, 2, 3, 4, 5],
# #               # [5, 6, 7, 8, 5],
# #               # [9, 10, 11, 12, 5],
# #               # [13, 14, 15, 16, 5]])
# z = CLs

# f = RegularGridInterpolator((x, y), z)
# z_approx = f([2400, 18])[0]



#%%
def interp2d(x, y, z, x_pred, y_pred, value, kind='linear'):
    if kind != 'linear':
        raise ValueError("Only linear interpolation is supported.")
    
    Cl = 0
    Cd = 0
    
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # Ensure x, y, and z have compatible shapes
    if x.shape[0] != z.shape[1] or y.shape[0] != z.shape[0]:
        raise ValueError("The shapes of x, y, and z must be compatible.")

    # Find indices of the nearest points in the grid
    i = np.searchsorted(x, x_pred) - 1
    j = np.searchsorted(y, y_pred) - 1

    # Check if the predicted point is within the range of the input data
    if i < 0 or i >= len(x) - 1 or j < 0 or j >= len(y) - 1:
        # Handle out of range prediction
        #print("Warning: Prediction point is out of range. Extrapolating...")
        
        # if value == 'Cl':
        #     print('entered C')
        #     if x_pred> max(x): 
        #         print('Entered > xpred>max(x)')
        #         Cl = Cd = 0
        #         c = 0
        #         to_test = np.linspace(max(x), x_pred,5)
        #         while Cl == 0 and Cd == 0 and c<5:
        #             print('looping: ', c)
        #             c+=1
        #             Cl, Cd = call_Xfoil(to_test[5-c], y_pred, aerofoil_name)
        #         if Cl !=0:
        #             print('Success Cl found')
        #             return Cl
        #     elif x_pred < min(x):
        #         print('Entered > xpred<min(x)')
        #         Cl = Cd = 0
        #         c = 0
        #         to_test = np.linspace(x_pred, min(x),5)
        #         while Cl == 0 and Cd == 0 and c<5:
        #             c+=1
        #             Cl, Cd = call_Xfoil(to_test[c], y_pred, aerofoil_name)
        #         if Cl != 0:
        #             return Cl
        #     if y_pred> max(y): #
        #         print('Entered > ypred>max(y)')
        #         Cl = Cd = 0
        #         c = 0
        #         to_test = np.linspace(max(y), y_pred,5)
        #         while Cl == 0 and Cd == 0 and c<5:
        #             c+=1
        #             Cl, Cd = call_Xfoil(x_pred, to_test[5-c], aerofoil_name)
        #         if Cl != 0:
        #             return Cl
        #     elif y_pred < min(y):
        #         print('Entered > ypred<min(x)')
        #         Cl = Cd = 0
        #         c = 0
        #         to_test = np.linspace(y_pred, min(y),5)
        #         while Cl == 0 and Cd == 0 and c<5:
        #             c+=1
        #             Cl, Cd = call_Xfoil(x_pred, to_test[c], aerofoil_name)
        #         if Cl != 0:
        #             print('success xfoil link')
        #             print(Cl)
        #             return Cl
            
        # if value == 'Cd':
        #     if x_pred> max(x): #
        #         print('Entered > xpred>max(x)')
        #         Cl = Cd = 0
        #         c = 0
        #         to_test = np.linspace(max(x), x_pred,5)
        #         while Cl == 0 and Cd == 0 and c<5:
        #             c+=1
        #             Cl, Cd = call_Xfoil(to_test[5-c], y_pred, aerofoil_name)
        #         if Cd !=0:
        #             return Cd
        #     elif x_pred < min(x):
        #         print('Entered > xpred<min(x)')
        #         Cl = Cd = 0
        #         c = 0
        #         to_test = np.linspace(x_pred, min(x),5)
        #         while Cl == 0 and Cd == 0 and c<5:
        #             c+=1
        #             Cl, Cd = call_Xfoil(to_test[c], y_pred, aerofoil_name)
        #         if Cd != 0:
        #             return Cd
        #     if y_pred> max(y): #
        #         print('Entered > ypred>max(y)')
        #         Cl = Cd = 0
        #         c = 0
        #         to_test = np.linspace(max(y), y_pred,5)
        #         while Cl == 0 and Cd == 0 and c<5:
        #             c+=1
        #             Cl, Cd = call_Xfoil(x_pred, to_test[5-c], aerofoil_name)
        #         if Cd != 0:
        #             return Cd
        #     # elif y_pred < min(y):
        #     #     print('Entered > ypred<min(y)')
        #     #     Cl = Cd = 0
        #     #     c = 0
        #     #     to_test = np.linspace(y_pred, min(y),5)
        #     #     while Cl == 0 and Cd == 0 and c<5:
        #     #         c+=1
        #     #         Cl, Cd = call_Xfoil(x_pred, to_test[c], aerofoil_name)
        #     #     if Cd != 0:
        #     #         print('success xfoil link')
        #     #         print(Cd)
        #     #         return Cd
            
        
        # if Cl == 0 or Cd == 0 or y_pred<min(y):
            
        
        # Perform linear extrapolation using the nearest points
        i = np.clip(i, 0, len(x) - 2)  # Ensure i is within bounds
        j = np.clip(j, 0, len(y) - 2)  # Ensure j is within bounds

    # Bilinear interpolation
    x1, x2 = x[i], x[i + 1]
    y1, y2 = y[j], y[j + 1]
    z11, z12 = z[j, i], z[j, i + 1]
    z21, z22 = z[j + 1, i], z[j + 1, i + 1]

    f_x1 = ((x2 - x_pred) / (x2 - x1)) * z11 + ((x_pred - x1) / (x2 - x1)) * z12
    f_x2 = ((x2 - x_pred) / (x2 - x1)) * z21 + ((x_pred - x1) / (x2 - x1)) * z22

    interpolated_value = ((y2 - y_pred) / (y2 - y1)) * f_x1 + ((y_pred - y1) / (y2 - y1)) * f_x2

    return interpolated_value


#%%
import numpy as np
N=10
dr = (0.1-0.02)/10

test_range = np.linspace(0.02 +(dr/2), 0.1-(dr/2), N)
#%%

starting_radius = 0.025
swept_radius = 0.1

def BEM(ga_instance, solution, solution_idx_):
    
    N =elements
    cord_length = solution[:N]
    twist = solution[N:2*N]
    TSR = solution[-1]
    
    
    N = elements
    mu = 1.81*1e-5
    rho = 1.204
    R=swept_radius
    B = Number_Blades
    a = np.zeros(N)
    
    a_prime = np.zeros(N)
    tol = 1e-2
    
    #TSR = to be extracted from solution
    OMEGA = TSR*U/R
    
    a_copy = np.zeros(N)+10
    a_prime_copy = np.zeros(N)+10
    viscous = 'True'
    Cts = []
    Cns = []
    Vs = []
    
    #cord length to be extracted from solution 
    #cord_length = np.linspace(max_cord_length, 0.006, N)
    #Starting_radius = 0.025
    dr = (R-starting_radius)/N
    r = np.linspace(starting_radius+(dr/2), R-(dr/2), N)
    #dr = (R-Starting_radius)/50
    
    for i in range(0,N):
        a_prime[i] =0
        a[i] = 0
        counter=0
        print('twist: ', np.degrees(twist[i]))
        print('cord length: ', cord_length[i])
        print('TSR: ', TSR)
        print('Omega: ', OMEGA)
        print('Radius: ', r[i])
        print('-------------Element found -----------------------------------')
        P=0
        C=0
        Re=10000
        while abs(a[i]-a_copy[i])>10e-3 or abs(a_prime[i]-a_prime_copy[i])>10e-3 or Re<2300 or a[i]>0.5:
            C+=1
            #print(twist[i])
            a_copy[i] = a[i]
            a_prime_copy[i] = a_prime[i]
            print('-----------')
            print('element: ',i)
            print('a: ', a)
            print('a_prime: ', a_prime)
            phi = np.arctan(U*(1-a[i])/(OMEGA*r[i]*(1+a_prime[i]))) # correct
            alpha = phi - twist[i] ### change twist # correct
            alpha_deg = np.degrees(alpha) # correct
                     
            V = np.sqrt((U*(1-a[i]))**2+(OMEGA*r[i]*(1+a_prime[i]))**2) # correct
            
            mu = 1.81*1e-5
            rho = 1.204
            
            Re = rho*V*cord_length[i]/mu  ### change cord_length # correct
            print('Reynolds: ', Re)
            #CL, CD = Extrapolate(fCD, fCL, Re, alpha_deg) # correct
            CL = interp2d(Attacks,Reynolds, CLs, alpha_deg, Re, value = 'Cl')
            CD = interp2d(Attacks, Reynolds, CDs, alpha_deg, Re, value = 'Cd')
            
            Ct = CL*np.sin(phi) - CD*np.cos(phi)  # correct and Cn
            Cn = CL*np.cos(phi) + CD*np.sin(phi)  # works
            
            solidity = B*cord_length[i]/(2*m.pi*r[i]) # correct

            a[i] = solidity*Cn/(4*np.sin(phi)**2+solidity*Cn)
            a_prime[i] = solidity*Ct/(4*np.sin(phi)*np.cos(phi)-solidity*Ct)

            counter +=1

            if counter ==50:
                return -np.inf
                #counter =0
                print('Axial induction factor: ', a[i])
                print('Angular induction factor: ', a_prime[i])
                print('Solidity: ', solidity)
                print('Ct: ', Ct)
                print('Cn: ', Cn)
                print('CL: ', CL)
                print('Cd : ', CD)
                print('phi: ', phi)
                print('Latest reynolds: ',Re)
                
                print('Convergence issue +++++++++++++++++++++++++++++++++++++++++')
        print('successful conver')

            
        Cts.append(Ct)
        Cns.append(Cn)
        Vs.append(V)
    
    Vs = np.array(Vs)

    print('Vs: ',Vs)
    print('Cts: ', Cts)
    print('Cns: ', Cns)
    print('Cord length: ', cord_length)
    print(' radius: ', r)
    
    delta_Q = 0.5*rho*Vs**2*B*cord_length*Cts*r*dr

    
    Cp = Cp_Calc(delta_Q, OMEGA, a, U, swept_radius)
    if Cp < 0:
        return -np.inf
    return Cp


def GA_mother(elements, U, max_cord, min_cord, max_twist, min_twist, TSR_max,\
              TSR_min, Number_Blades, aerofoil_name):
    GS = []
    twists = []
    
    for n in range(elements):
        new_dict = {'low': min_cord, 'high': max_cord}
        GS.append(new_dict)
        new_dict = {'low': min_twist, 'high': max_twist}
        twists.append(new_dict)
    
    GS.extend(twists)
    GS.append({'low': TSR_min, 'high': TSR_max})
    
    #gene_space = {}
    #extrapolate_map create
    
    #pre defining GA properties.
    num_generations = 30
    num_parents_mating = 40

    fitness_function = BEM

    sol_per_pop = 200
    num_genes = elements*2+1

    # init_range_low = 40
    # init_range_high = 50

    parent_selection_type = "sss"
    keep_parents = 30

    crossover_type = "single_point"

    mutation_type = "random"
    mutation_percent_genes = 50
    def on_gen(ga_instance):
        print("Generation : ", ga_instance.generations_completed)
        print("Fitness of the best solution :", ga_instance.best_solution()[1])

    # this GS must be sized appropriately to account for all the parameters. 
    # GS = [{'low': 0.68, 'high': 5.8},
    #       {'low': 6, 'high': 8},
    #       {'low': 40, 'high': 50},
    #       {'low': 50, 'high': 60}
    #       ]

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           # init_range_low=init_range_low,
                           # init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           gene_space = GS)

    ga_instance.run()

    ga_instance.plot_fitness()

    solution_, solution_fitness, solution_idx_LiBr = ga_instance.best_solution()
    
    return solution_

elements=10
U=3
max_cord=0.03
min_cord=0.007
max_twist = m.pi/2
min_twist=0
TSR_max = 10
TSR_min = 2
Number_Blades = 3



solution_  =GA_mother(elements, U, max_cord, min_cord, max_twist, min_twist, TSR_max, TSR_min, Number_Blades, aerofoil_name)


R = swept_radius
dr = (swept_radius-starting_radius)/elements
r = np.linspace(starting_radius+(dr/2), R-(dr/2), elements)
optimal_cord_length = solution_[:elements]
optimal_twist = solution_[elements:2*elements]
optimal_TSR = solution_[-1]

plt.plot(r, optimal_twist)
plt.xlabel('Distance from Hub (m)')
plt.ylabel('Optimal_twist angle (Radians)')
plt.title('Twist (rad) vs. Distance from Hub (m)')

plt.show()

plt.plot(r, optimal_cord_length)
plt.xlabel('Distance from Hub (m)')
plt.ylabel('Cord Lengths (Meters)')
plt.title('Cord Length (m) vs. Distance from Hub (m)')

plt.show()
print('Optimal TSR: ', optimal_TSR)
