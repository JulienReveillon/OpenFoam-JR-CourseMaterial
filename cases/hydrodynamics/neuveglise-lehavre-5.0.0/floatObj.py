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


d=0.6

#======================Defining the directories==========================

pathname = os.path.abspath('.')
samplingDir = os.path.join(pathname,'olaDyMFlow.log')
waveDir = os.path.join(pathname,'constant/waveDict')
controlDir = os.path.join(pathname,'system/controlDict')

#================Parameters (H,T,d,L,rho,tBegin,tEnd,tStep)================

tBegin = 0 #float(raw_input("Please enter the start time(s) tBegin: "))
tEnd   = 120 #float(raw_input("Please enter the end time(s) tEnd: "))

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
                   z1=z1[:-1]
                   z[n-1]=float(z1) 
                else:
                   z1=word[5]
                   z1 = z1[:-1]
                   z.append(float(z1))


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
#=============================================================
gs1 = gridspec.GridSpec(1, 1)
gs1.update(hspace=0.3, wspace=0.3)

fig = plt.figure()

sub2 = plt.subplot(gs1[0, :])
sub2 = plt.plot(t,z,'k')
plt.axis([tBegin,tEnd,minZ-0.05,maxZ+0.05])
plt.xlabel('time (s)')
plt.ylabel('Zcm (m)')
plt.grid(True)

plt.savefig('postProcess.png')
plt.show()
plt.close

file = open('Zobj.txt','w')
np.savetxt(file, z,fmt=['%.6f'])
file.close() 

file = open('time.txt','w')
np.savetxt(file, t,fmt=['%.6f'])
file.close() 

print ('x..............................................................x')
print ('x...             Post-pro Over                              ...x')
print ('x..............................................................x')
