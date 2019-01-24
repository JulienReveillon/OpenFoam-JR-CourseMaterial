#!/usr/bin/python
import os, sys

print('x..............................................................x')
print('x...             OpenFOAM - www.foam-u.fr                   ...x')
print('x..............................................................x')
# ==================================
# ==> Python Module Importations <==
# ==================================
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from   matplotlib.backends.backend_pdf import PdfPages

# ================Definition Parameters================
fetch =input('Please enter the fetch (m)\n')   # 

u10 =input('Please enter the u10 (m/s)\n')    # Velocity of the wind 10m above from the sea level
 
g=9.81         # Gravity Force

#Tp = 1.58     # Peak period

gamma = 3.3    # 3.3 common value

tEnd = 50    # Studied time

rho = 1000     # density of the water (kg/m3)

#======================================================

alpha = 0.076*np.power((g*fetch/np.power(u10,2)),-0.22)

step = 1.0/tEnd 

# Use one or the other 

#fp=1.0/Tp

fp = (3.5*np.power(np.power(g,2.0)/(fetch*u10),0.33))

print('fp')
print(fp)

wp=2.0*np.pi*fp

def S(f):
   
#   w=2.0*np.pi*f

   if np.all(f<=fp):
     r1 = np.exp  (-(1.0/2)  *  np.power((f-fp)/(fp*0.07),2.0)  )
     return ((alpha*np.power(g,2.0)) / ((np.power(2.0*np.pi,4.0))*(np.power(f,5.0)))) * np.exp(-(5.0/4)*np.power((fp/f),4.0)) * np.power(gamma,r1)

   else :
     r2 = np.exp  (-(1.0/2)  *  np.power((f-fp)/(fp*0.09),2.0)  )
     return ((alpha*np.power(g,2.0)) / ((np.power(2.0*np.pi,4.0))*(np.power(f,5.0)))) * np.exp(-(5.0/4)*np.power((fp/f),4.0)) * np.power(gamma,r2)


t1 = np.arange(step, 10, step)
t2=[]
for index, item in enumerate(t1):

  if item > fp:
    if S(item) > S(fp)*0.05:
       t2.append(item)
  elif item <= fp:
    if S(item) > S(fp)*0.01:
       t2.append(item)
        

H=[]
T=[]
phase=[]  
mo=0

for l in t2:

      mo=step*S(l)+mo
      H.append((np.power(2*S(l)*step*np.pi,0.5))) 
      T.append(1.0/l)


Hmo=4*np.power(mo,0.5) # Moment zero wave height

print('Hmo:')
print(Hmo)

print('S(fp):')
print(S(fp))

Hmax=max(H)
Smax=max(S(t1))
t2max=max(t2)
n1=len(H)


phase = np.random.uniform(low=(-np.pi), high=np.pi, size=n1) # Random phase
directions=np.random.uniform(low=0, high=0, size=n1)

maxPhase=max(phase)


gs1 = gridspec.GridSpec(2, 2)
gs1.update(hspace=0.3, wspace=0.3)

fig = plt.figure()

plt.title('Jonswap Spectrum F=1 km U10=5 m/s gamma = 3.3')

sub1 = plt.subplot(gs1[:-1, :])

sub1=plt.plot(t2,S(t2),'k', ms=4)
plt.axis([0,t2max+0.3*t2max , 0, Smax+0.3*Smax])
plt.grid(True)
plt.xlabel('Frequency (Hz)')
plt.ylabel('S(f) (m2s)')


sub2 = plt.subplot(gs1[-1, :-1])

sub2=plt.plot(t2,H,'k.', ms=4)
plt.axis([0,t2max+0.3*t2max , 0, Hmax+0.3*Hmax])
plt.xlabel('Frequency (Hz)')
plt.ylabel('H (m)')
plt.grid(True)

sub3 = plt.subplot(gs1[-1, -1])

sub2=plt.bar(t2,phase,width=0.002, color="black")
plt.axis([0,t2max+0.3*t2max , -maxPhase-0.3*maxPhase, maxPhase+0.3*maxPhase])
plt.xlabel('Frequency (Hz)')
plt.ylabel('phase (rad)')
plt.grid(True)


plt.show()

pp = PdfPages('input_spectrum.pdf')
pp.savefig(fig)
pp.close()

#===========Writing the waveDict file==================

file = open('constant/waveDict','a') 

file.write('wavePeriods\n')

file.write('{0}\n'.format(n1))

file.write('(\n')

np.savetxt(file, T,fmt=['%.6f'])

file.write(');\n')

file.write('waveHeights\n')

file.write('{0}\n'.format(n1))

file.write('(\n')

np.savetxt(file, H, fmt=['%.6f'])

file.write(');\n')

file.write('wavePhases\n')

file.write('{0}\n'.format(n1))

file.write('(\n')

np.savetxt(file, phase, fmt=['%.6f'])

file.write(');\n')

file.write('waveDirs\n')

file.write('{0}\n'.format(n1))

file.write('(\n')

np.savetxt(file, directions, fmt=['%.6f'])

file.write(');\n')

file.close() 

print ('x..............................................................x')
print ('x...             Pre-pro  Over                              ...x')
print ('x..............................................................x')
