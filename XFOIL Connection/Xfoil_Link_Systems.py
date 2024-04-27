# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 10:46:39 2024

@author: jrjol
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 23:47:33 2024

@author: jrjol
"""

import subprocess
import os 
import numpy as np

def call_Xfoil(alpha, Re, airfoil_name, viscous):
    
    
    if viscous == 'True':
        
        n_iter=200
        if os.path.exists("polar_file.txt"):
            os.remove("polar_file.txt")
        #print('airfoil: ', airfoil_name)
        input_file = open("input_file.in", 'w')
        input_file.write("{0}\n".format(airfoil_name))
        #input_file.write(airfoil_name + '\n')
        input_file.write("PANE\n")
        input_file.write("OPER\n")
        input_file.write("Visc {0}\n".format(Re))
        input_file.write("PACC\n")
        input_file.write("polar_file.txt\n\n")
        input_file.write("ITER {0}\n".format(n_iter))
        input_file.write("Alfa {0}\n".format(alpha))
        input_file.write("\n\n")
        input_file.write("quit\n")
        input_file.close()
    
        subprocess.call("xfoil < input_file.in", shell=True) # call xfoil and input all info
        #polar_file_path = r'C:\Users\jrjol\Documents\XFoil\XFOIL6.99\polar_file.txt'
        # #polar_data = np.loadtxt(polar_file_path, skiprows=12)
        # try:
        #     subprocess.call("xfoil.exe < input_file.in", shell=True)
        # except Exception as e:
        #     print(f"Error executing XFOIL: {e}")
        polar_data = np.loadtxt("polar_file.txt", skiprows=12)
        if len(polar_data) == 0:
            return 0,0
            
        Cl = polar_data[1]
        Cd = polar_data[2]
        return Cl, Cd
    else:
        n_iter=200
        if os.path.exists("polar_file.txt"):
            os.remove("polar_file.txt")
        #print('airfoil: ', airfoil_name)
        input_file = open("input_file.in", 'w')
        input_file.write("{0}\n".format(airfoil_name))
        #input_file.write(airfoil_name + '\n')
        input_file.write("PANE\n")
        input_file.write("OPER\n")
        #input_file.write("Visc {0}\n".format(Re))
        input_file.write("PACC\n")
        input_file.write("polar_file.txt\n\n")
        input_file.write("ITER {0}\n".format(n_iter))
        input_file.write("Alfa {0}\n".format(alpha))
        input_file.write("\n\n")
        input_file.write("quit\n")
        input_file.close()
    
        subprocess.call("xfoil < input_file.in", shell=True) # call xfoil and input all info
        #polar_file_path = r'C:\Users\jrjol\Documents\XFoil\XFOIL6.99\polar_file.txt'
        # #polar_data = np.loadtxt(polar_file_path, skiprows=12)
        # try:
        #     subprocess.call("xfoil.exe < input_file.in", shell=True)
        # except Exception as e:
        #     print(f"Error executing XFOIL: {e}")
        polar_data = np.loadtxt("polar_file.txt", skiprows=12)
        if len(polar_data) == 0:
            return 0,0
            
        Cl = polar_data[1]
        Cd = polar_data[2]
        return Cl, Cd
    
Cl, Cd = call_Xfoil(5, 4000, "naca 4415", viscous = 'True')

