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
samplingDir = os.path.join(pathname,'olaDyMFlow.log')
waveDir = os.path.join(pathname,'constant/waveDict')
controlDir = os.path.join(pathname,'system/controlDict')

#================Parameters (H,T,d,L,rho,tBegin,tEnd,tStep)================

tBegin = 0
tEnd   = 50	
stiffness = 100
l0 = 1

#======================Numerical data treatment==========================

graphHeight = []
t   	= []
SL   	= []
J	= []
n1=0

with open(samplingDir) as openfile:
    for line in openfile:
        for part in line.split():
            if "deltaT" in part:
                word=line.split()
                t.append(n1+float(word[2]))
                n=len(t)
                n1=t[n-1]
            if "spring:" in part:
                word=line.split()
                if len(SL)==n:
                   sl=word[10]
                   SL[n-1]=float(sl)                   
                else:
                   sl=word[10]
                   SL.append(float(sl))

   
J = [ 0.5*stiffness*np.power(float(x)-l0,2.0) for x in SL]

minSL= min(SL)
maxSL= max(SL)

maxJ= max(J)
minJ= min(J)

#=============================================================
gs1 = gridspec.GridSpec(1, 1)
gs1.update(hspace=0.3, wspace=0.3)

fig = plt.figure()

#sub1 = plt.subplot(gs1[:-1, :])
#sub1=plt.plot(t,SL,'k')
#plt.axis([tBegin,tEnd,minSL-0.05,maxSL+0.05])
#plt.xlabel('time (s)')
#plt.ylabel('Spring length (m)')
#plt.grid(True)

sub2 = plt.subplot(gs1[-1, :])
sub2 = plt.plot(t,J,'k')
plt.axis([tBegin,tEnd,minJ-100,maxJ+100])
plt.xlabel('time (s)')
plt.ylabel('Energy (J)')
plt.grid(True)

plt.savefig('Energy.png')
plt.show()
plt.close

file = open('Spring_length.txt','w')
np.savetxt(file, SL,fmt=['%.6f'])
file.close() 
file = open('Energy.txt','w')
np.savetxt(file, J,fmt=['%.6f'])
file.close() 
file = open('time.txt','w')
np.savetxt(file, t,fmt=['%.6f'])
file.close() 

print ('x..............................................................x')
print ('x...             Post-pro Over                              ...x')
print ('x..............................................................x')
