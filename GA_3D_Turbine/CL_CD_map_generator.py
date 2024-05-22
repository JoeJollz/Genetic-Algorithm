# -*- coding: utf-8 -*-
"""
This code creates CL and CD maps for a range of alphas and reynolds. 

RegularGrid and NearestNeighbour Interpolators from Scipy were used and extracted
into .pkl files. 

Evolutionary Strategies and Tabu Search algorithms use the RegularGrid interpolator.

Genetic Algorithm use the NearestNeighbour Interpolator.

^ This was due to personal preference within our grid, each interpolator has
its weaknesses, especially when working out of the alpha and Re range, for 
which the map was designed for. 

This code has a long computational time, you do not need to run it as you can
download all the .pkl files from our submission space.

We create interpolation maps for the following aerofoils (allowing for our 
    optimisers to pick the most appropriate aerofoil for this project):
    
    NACA 4412
    NACA 4415
    NACA 23012

"""

import subprocess
import numpy as np
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import RegularGridInterpolator
import pickle


def call_Xfoil(alpha, Re, airfoil_name):
    '''
    Parameters
    ----------
    alpha : FLOAT
        Input the angle of attack, in degrees.
    Re : FLOAT
        Input the Reynolds number.
    airfoil_name : STRING
        Input the aerofoil name.

    Returns
    -------
    Cl : FLOAT
        Coefficient of lift.
    Cd : FLOAT
        Coefficient of drag.
    '''

    with open('xfoilinput.txt', 'w') as out:

        out.write(
            f"""
            {airfoil_name}
            panel
            OPER
            iter 100
            visc
            {Re}
            A {alpha}
            !

            quit
            """
        )

    xfoil_process = subprocess.Popen(['xfoil'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    stdout, stderr = xfoil_process.communicate(input=open('xfoilinput.txt').read())

    if xfoil_process.poll() is None:
        xfoil_process.kill()
        print("Xfoil was forcefully terminated due to timeout.")
        return 0, 0

    cl, cd = parse_output(stdout)
    
    print('CL: ', cl)
    print('Cd: ', cd)
    return cl, cd

def parse_output(output):
    '''
    This function reads the output from xfoil, and attempts to identify the 
    converged CL and CD.
    
    Parameters
    ----------
    output : TYPE
        The xfoil file output.

    Returns
    -------
    Cl_value : FLOAT
        Coefficient of lift.
    Cd_value : FLOAT
        Coefficient of drag.

    '''
    cl_value = 0
    cd_value = 0

    lines = output.split('\n')
    for line in reversed(lines):
        if 'CL =' in line and cl_value == 0:
            cl_index = line.index('CL =') + 5
            cl_str = line[cl_index:].strip()
            if '********' in cl_str:
                return 0, 0  # Return 0, 0 if '********' is encountered
            cl_value = float(cl_str)

        elif 'CD =' in line and cd_value == 0:
            cd_index = line.index('CD =') + 5
            cd_str = line[cd_index:].strip()
            if '********' in cd_str:
                return 0, 0  # Return 0, 0 if '********' is encountered
            cd_value = float(cd_str.split()[0])  

    return cl_value, cd_value


aerofoils = ['naca4412', 'naca4415', 'naca23012']

Reynolds = np.array([ 2500,5000, 10000, 20000,30000, 40000,50000, 60000, 70000, 80000, 160000, 320000, 640000])

Attacks = np.linspace(-7,21, 20) 

CL_interpolators = {}
CD_interpolators = {}

CL_interpolators_GA = {}
CD_interpolators_GA = {}

for aerofoil_name in aerofoils:
    
    hashmap_Cd = {}
    hashmap_Cl = {}
    
    X, Y = np.meshgrid(Reynolds,Attacks)
    CLs = np.empty((0, len(Attacks)))
    CDs = np.empty((0, len(Attacks)))
    for Re in Reynolds: # y-axis
        piecewise = 'False'
        CL=[]
        CD=[]
        for i, attack in enumerate(Attacks): # x-axis
            print('-------------------------------')
            print('aerofoil: ', aerofoil_name)
            print('run: ', i)
            print('Reynolds: ', Re)
            print('attack: ', attack)
            
            attack = np.round(attack, 2)
            
            if Re<2300:
                
                Cl = Cd = 0
            else:
                Cl, Cd = call_Xfoil(attack, Re, airfoil_name=aerofoil_name)
                Re_ = Re
                
                while Cl ==0 and attack>4:
                    Re_=Re_*0.98
                    Cl, Cd = call_Xfoil(attack, Re_, airfoil_name=aerofoil_name)
            CL.append(Cl)
            CD.append(Cd)
            
        CL = np.array(CL)
        CD = np.array(CD)

        CLs = np.vstack((CLs, CL))
        CDs = np.vstack((CDs, CD))
            
    A, R = np.meshgrid(Attacks, Reynolds)
    
    points = np.array([A.flatten(), R.flatten()]).T  # Shape (N, 2)
    values_CL = CLs.flatten()  # Shape (N,)
    values_CD = CDs.flatten()
    
    CL_interpolators[aerofoil_name] = RegularGridInterpolator((Reynolds, Attacks), CLs, method='linear')
    CD_interpolators[aerofoil_name] = RegularGridInterpolator((Reynolds, Attacks), CDs, method='linear')
    CL_interpolators_GA[aerofoil_name] = NearestNDInterpolator(points, values_CL)
    CD_interpolators_GA[aerofoil_name] = NearestNDInterpolator(points, values_CD)
    

with open('CL_interpolators.pkl', 'wb') as f:

    pickle.dump(CL_interpolators, f)
 
with open('CD_interpolators.pkl', 'wb') as f:

    pickle.dump(CD_interpolators, f)

with open('CL_interpolators_GA.pkl', 'wb') as f:

    pickle.dump(CL_interpolators_GA, f)
 
with open('CD_interpolators_GA.pkl', 'wb') as f:

    pickle.dump(CD_interpolators_GA, f)