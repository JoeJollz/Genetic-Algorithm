# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 11:35:41 2024

@author: jrjol
"""

from io import *
import os
import re

#open the file for the xfoil commands
out=open('xfoilinput.txt', 'w')

# Write xfoil commands to the file (as they would 
# be typed into xfoil)
out.write(
"""
naca 4415
panel
OPER
iter 200
visc
A 5
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
    cl = matches.group(1)
    print("Lift coefficient =", cl)
else:
    cl = 0
    print("CL not found")

# Search for CD value
matches = re.search(cd_pattern, text[6])
if matches:
    cd = matches.group(1)
    print("Drag coefficient =", cd)
else:
    cd = 0
    print("CD not found")
