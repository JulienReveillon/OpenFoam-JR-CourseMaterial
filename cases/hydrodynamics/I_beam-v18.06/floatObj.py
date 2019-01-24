#!/usr/bin/env python3

print('x..............................................................x')
print('x...             OpenFOAM - www.foam-u.fr                   ...x')
print('x..............................................................x')
# ==================================
# ==> Python Module Importations <==
# ==================================
import os,sys,shutil, fileinput
from pylab import *
from scipy import interpolate
import numpy as np
import matplotlib.gridspec as gridspec
from   matplotlib.backends.backend_pdf import PdfPages
from scipy import fft, arange
import matplotlib.pyplot as plt
import cmath

#======================Defining the directories==========================

pathname = os.path.abspath('.')
samplingDir = os.path.join(pathname,'background/log.overInterDyMFoam')
controlDir = os.path.join(pathname,'background/system/controlDict')

#================Parameters (H,T,d,L,rho,tBegin,tEnd,tStep)================

tBegin = 0 #float(raw_input("Please enter the start time(s) tBegin: "))
tEnd   = 60 #float(raw_input("Please enter the end time(s) tEnd: "))

rhoW = 1000 # Change this value if needed         

d = 5

# Searching the values of startTime, endTime and writeInterval in controlDict file

with open(controlDir) as openfile:
    for line in openfile:
        for part in line.split():
            if "writeInterval" in part:
                word=line.split()
                tStep=word[1]
                tStep = tStep[:-1]
                tStep=float(tStep)  
 

#======================Numerical data treatment==========================



graphHeight = []
t	= []
z	= []
x	= []
beta	= []
n1=0

with open(samplingDir) as openfile:
    for line in openfile:
        for part in line.split():
            if "deltaT" in part:
                word=line.split()
                t.append(n1+float(word[2]))
                n=len(t)
                n1=t[n-1]
            if "rotation:" in part:
                word=line.split()
                if len(z)==n:
                   z1=word[5]
                   x1=word[3]
                   z1=z1[:-1]
                   x1=x1[1:-1]
                   z[n-1]=float(z1) 
                   x[n-1]=float(x1)                    
                else:
                   z1=word[5]
                   x1=word[3]
                   z1 = z1[:-1]
                   x1=x1[1:-1]
                   z.append(float(z1))
                   x.append(float(x1))
            if "Orientation:" in part:
            	word=line.split()
            	if len(beta)==n:
                   b1=word[3]
                   beta[n-1]=b1                   
                else:
                   b1=word[3]
                   beta.append(float(b1))
   
x[0]=10
beta = [ np.arcsin(float(k))*(180/np.pi) for k in beta]

indexZ=[]
for index, item in enumerate(z):
    if item > d*2:

       indexZ.append(index)
    if item < 0:

       indexZ.append(index)

t=np.delete(t, indexZ)
z=np.delete(z, indexZ)

minZ= min(z)
maxZ= max(z)
minX= min(x)
maxX= max(x)
minB= min(beta)
maxB= max(beta)
#=============================================================
gs1 = gridspec.GridSpec(3, 3)
gs1.update(hspace=0.3, wspace=0.3)

fig = plt.figure()

sub1 = plt.subplot(gs1[0, :])
sub1=plt.plot(t,beta,'k')
plt.axis([tBegin,tEnd,minB-5,maxB+5])
plt.xlabel('time (s)')
plt.ylabel('Beta (deg)')
plt.grid(True)

sub2 = plt.subplot(gs1[1, :])
sub2 = plt.plot(t,z,'k')
plt.axis([tBegin,tEnd,minZ-0.05,maxZ+0.05])
plt.xlabel('time (s)')
plt.ylabel('Zcm (m)')
plt.grid(True)

sub3 = plt.subplot(gs1[-1, :])
sub3 = plt.plot(t,x,'k')
plt.axis([tBegin,tEnd,minX-0.05,maxX+0.05])
plt.xlabel('time (s)')
plt.ylabel('X (m)')
plt.grid(True)

plt.savefig('postProcess.png')
plt.show()
plt.close

file = open('Zobj.txt','w')
np.savetxt(file, z,fmt=['%.6f'])
file.close() 
file = open('Xobj.txt','w')
np.savetxt(file, x,fmt=['%.6f'])
file.close() 
file = open('Beta.txt','w')
np.savetxt(file, beta,fmt=['%.6f'])
file.close() 
file = open('time.txt','w')
np.savetxt(file, t,fmt=['%.6f'])
file.close() 

print ('x..............................................................x')
print ('x...             Post-pro Over                              ...x')
print ('x..............................................................x')
