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
from   matplotlib.backends.backend_pdf import PdfPages
from scipy import fft, arange
import matplotlib.pyplot as plt
import cmath

#======================Defining the directories==========================

pathname = os.path.abspath('.')
samplingDir = os.path.join(pathname,'background/postProcessing/sampleDict/')
waveDir = os.path.join(pathname,'background/constant/waveDict')
controlDir = os.path.join(pathname,'background/system/controlDict')

#================Parameters (H,T,d,L,rho,tBegin,tEnd,tStep)================

d = float(raw_input("Please enter depth(m) d: "))

rho = 1000 # Change this value if needed 

# Searching the values of the wave height and period in the waveDict file

waveType=""
with open(waveDir) as openfile:

    for line in openfile:
        for part in line.split():
            if "waveType" in part:
                word=line.split()
                waveType=word[1]
                waveType = waveType[:-1]
with open(waveDir) as openfile:  
    if waveType=="irregular":
        waveDict=openfile.readlines()

        H=[]
        T=[]
        n0=float(waveDict[28])
        waveDict = map(str.strip, waveDict) # Remove the \n from each component of the list
        n1=int(30+n0)                     # Index of the last T data
        n2=int(n1+4)                      # Index of the first H data                      
        n3=int(n2+n0)                     # Index of the last H data
        T=waveDict[30:n1]
        H=waveDict[n2:n3]

        H = list(map(float, H))
        T = list(map(float, T))
    elif waveType=="regular":
         for line in openfile:
             for part in line.split():
                 if "waveTheory" in part:
                     word=line.split()                   
                     waveTheory=word[1]
                     waveTheory = waveTheory[:-1]
                 if "waveHeight" in part:
                     word=line.split()
                     H=word[1]
                     H = H[:-1]
                     H=float(H)
                 if "wavePeriod" in part:
                     word=line.split()
                     T=word[1]
                     T = T[:-1]
                     T=float(T)         

# Searching the values of startTime, endTime and writeInterval in controlDict file

with open(controlDir) as openfile:
    for line in openfile:
        for part in line.split():
            if "startTime" in part:
                word=line.split()
                tBegin=word[1]
                tBegin = tBegin[:-1]
                tBegin=float(tBegin)
            if "endTime" in part:
                word=line.split()
                if "endTime"==word[0]:
                  tEnd=word[1]
                  tEnd = tEnd[:-1]
                 # tEnd = 120 # Fix here if your tEnd doesn't fit with the controlDict
                  tEnd=float(tEnd)
                else:continue
            if "writeInterval" in part:
                word=line.split()
                tStep=word[1]
                tStep = tStep[:-1]
                tStep=float(tStep)  
   

# Wave Lenght=====================================

def waveNumHunt(T, h):
    if type(T) == list:
        T = np.array(T)
    if type(h) == list:
        h = np.array(h)       
    
    D = np.array([0.6666666666, 0.3555555555, 0.1608465608, 0.0632098765, 0.0217540484, 0.0065407983])
    
    L0 = 9.81 * T**2 / (2.*np.pi)
    k0 = 2. * np.pi / L0
    k0h = k0 * h
    
    aux = D[0] * k0h + D[1] * k0h**2 + D[2] * k0h**3 + D[3] * k0h**4 + D[4] * k0h**5 + D[5] * k0h**6
    
    kh = k0h * np.sqrt( 1. + ( k0h * ( 1. + aux ))**-1  )
    
    k = kh / h
    L = 2. * np.pi / k
    return L

L = waveNumHunt(T, d)
if waveType=="regular":

   print("L :")
   print(L)
#=================================================
# Frequency

frequency=[] 
Ho=[]

if waveType=="regular":
   # General constants

   w=2*np.pi/T

   k=(2*np.pi/L)

   kd=k*d
 
   if waveTheory=="StokesI":
      Ho.append(H/2)
      frequency.append(1.0/T)
   elif waveTheory=="StokesII":
      Ho.append(H/2)
      Ho.append(k*((np.power(H,2))/4)*((3-np.power(np.tanh(kd),2))/(4*(np.power(np.tanh(kd),3)))))
      frequency.append(1.0/T)
      frequency.append(2.0/T) 
elif waveType=="irregular":
   Ho=[ x/2.0 for x in H ]

   frequency=[ 1.0/x for x in T ]


#======================Numerical data treatment==========================

time = np.arange(tBegin,tEnd,tStep)

graphTime = []
graphHeightG1 = []
graphHeightG2 = []
graphHeightG3 = []
index=[]

for t in time:

    tRound  = float(math.floor(t))
    if t - tRound < tStep-0.3*tStep:
        cTime =str(int(math.floor(t)))
    else:
        cTime = str(t)
    timeSegmentDir      = samplingDir+cTime+'/'
    segmentFileG1         = timeSegmentDir+'lineX1_alpha.water.xy' 
    segmentFileG2         = timeSegmentDir+'lineX2_alpha.water.xy'
    segmentFileG3         = timeSegmentDir+'lineX3_alpha.water.xy'
    #print(segmentFileG1)
    if not os.path.isfile(segmentFileG1):
        print ('E... '+'no lineFile')
        sys.exit()
    z       = []
    alpha   = []

    for line in fileinput.input(segmentFileG1):
        words = line.split()
        if words[0]=='#':
	        print (words)
        else:
            z.append(float(words[0]))     # We create the elevation vector 
            alpha.append(float(words[1])) # We create the alpha vector 

    f = interpolate.interp1d(z,alpha)
    z1=np.min(z)
    z2=np.max(z)
    while z2-z1 > 1e-5:
        z=0.5*(z2+z1)
        if f(z) > 0.5 : z1=z
        elif f(z) < 0.5 : z2 = z
	else : break
    z=0.5*(z2+z1)
    
    graphTime.append(t)
    graphHeightG1.append(z)
    z       = []
    alpha   = []    
    for line in fileinput.input(segmentFileG2):
        words = line.split()
        if words[0]=='#':
	        print (words)
        else:
            z.append(float(words[0]))     # We create the elevation vector 
            alpha.append(float(words[1])) # We create the alpha vector 

    f = interpolate.interp1d(z,alpha)
    z1=np.min(z)
    z2=np.max(z)
    while z2-z1 > 1e-5:
        z=0.5*(z2+z1)
        if f(z) > 0.5 : z1=z
        elif f(z) < 0.5 : z2 = z
	else : break
    z=0.5*(z2+z1)
    
    graphHeightG2.append(z)
    z       = []
    alpha   = []
    for line in fileinput.input(segmentFileG3):
        words = line.split()
        if words[0]=='#':
	        print (words)
        else:
            z.append(float(words[0]))     # We create the elevation vector 
            alpha.append(float(words[1])) # We create the alpha vector 

    f = interpolate.interp1d(z,alpha)
    z1=np.min(z)
    z2=np.max(z)
    while z2-z1 > 1e-5:
        z=0.5*(z2+z1)
        if f(z) > 0.5 : z1=z
        elif f(z) < 0.5 : z2 = z
	else : break
    z=0.5*(z2+z1)
    
    graphHeightG3.append(z)

maxH=[]
minH=[]
if waveType=="irregular":
   maxH.append(max(graphHeightG1))
   maxH.append(max(graphHeightG2))
   maxH.append(max(graphHeightG3))
   maxHo=max(maxH)
   minH.append(min(graphHeightG1))
   minH.append(min(graphHeightG2))
   minH.append(min(graphHeightG3))
   minHo=min(minH)
# ______________________Plot numerical solution_________________________

# Surface evolution in time

fig1=plt.figure()

sub1=plt.subplot(311)
sub1=plt.plot(graphTime,graphHeightG1,'k', ms=4, label='Numerical Approach')
plt.grid(True)
if waveType=="regular":
    plt.axis([tBegin, tEnd, d-(1.5*H), d+(1.5*H)])
elif waveType=="irregular":
    plt.axis([tBegin, tEnd, minHo-0.5, maxHo+0.5])
sub2=plt.subplot(312)
sub2=plt.plot(graphTime,graphHeightG2,'k', ms=4, label='Numerical Approach')
plt.grid(True)
if waveType=="regular":
    plt.axis([tBegin, tEnd, d-(1.5*H), d+(1.5*H)])
elif waveType=="irregular":
    plt.axis([tBegin, tEnd, minHo-0.5, maxHo+0.5])
sub3=plt.subplot(313)
sub3=plt.plot(graphTime,graphHeightG3,'k', ms=4, label='Numerical Approach')

if waveType=="regular":
    plt.axis([tBegin, tEnd, d-(1.5*H), d+(1.5*H)])
elif waveType=="irregular":
    plt.axis([tBegin, tEnd, minHo-0.5, maxHo+0.5])
plt.grid(True)
plt.show()


#====================Gauge=================================================

gauge =int(raw_input("Which gauge do you want to use, write the number: "))

# ===================Frequency-Amplitude Spectrum =========================

graphHeight=[]

if gauge==1:
   graphHeight=graphHeightG1
elif gauge==2:
   graphHeight=graphHeightG2
elif gauge==3:
   graphHeight=graphHeightG3
#================= Regular case =========================
if waveType=="regular":
 counter=0
 auxTime=np.arange(tBegin,tEnd,0.01)
 f1=interpolate.interp1d(time,graphHeight)   
 ind=0

 interT=[]
 interH=[]

 testH=[]
 testT=[]

 for index,item in enumerate(auxTime):
    interH.append(f1(item))
    interT.append(item)
    if f1(item)<(d) and f1(item+0.01)>(d) :

       testT.append(item)
       testH.append(f1(item))

       counter=counter+1

       if counter==70:
         aux1=item+0.005
         testT.append(item)
         testH.append(f1(item))
         break 

    if f1(item)>(d) and f1(item+0.01)<(d) :

       counter=counter+1

       testT.append(item)
       testH.append(f1(item))

       if counter==70:
         aux1=item+0.005
         testT.append(item)
         testH.append(f1(item))
         break 

 for index,item in enumerate(time):
    if item>=aux1:
      time1=time[index:]
      height1=graphHeight[index:]   
      break 

 f2=interpolate.interp1d(time1,height1)
 auxTime1=arange(min(time1),max(time1),0.01)
 height2=[]

 for index,item in enumerate(auxTime1):
    if (index+1)==(len(auxTime1)):
       break
    else: height2.append(f2(item))
 n=len(height2)

 frq=np.fft.fftfreq(n,tStep)       
 frq=frq[1:]
#================= Irregular case =========================
if waveType=="irregular": 
   
   height2=graphHeight
   height2=height2[100:]

   n=len(height2)

   frq=np.fft.fftfreq(n,tStep)      
   frq=frq[1:]




cO=np.fft.fft(height2)*2.0/n
        
cO=cO[1:]


index=[]
counter=0

if waveType=="regular":
   if waveTheory=="StokesI":
       for l in cO:
           if abs(l)<=((H/2.0)*0.01): # The minimum amplitude threshold we defined as (Amplitude of 1 order Stoke wave)*0.1
              index.append(counter)
           counter=counter+1
   elif waveTheory=="StokesII":
       for l in cO:
           if abs(l)<=((k*((np.power(H,2))/4)*((3-np.power(np.tanh(kd),2))/(4*(np.power(np.tanh(kd),3)))))*0.2): # The minimum amplitude threshold we defined as (Amplitude of 2 order Stoke wave)*0.1
              index.append(counter)
           counter=counter+1
elif waveType=="irregular":
   minH=min(H)
   maxH=max(H)
   for l in cO:
           if abs(l)<=((minH/2.0)*0.1):
              index.append(counter)
           counter=counter+1
            
frq = np.delete(frq, index)
cO = np.delete(cO, index)

maxfrq=max(frq)



#====================Plot free surface gauge choosen============================

fig5=plt.figure()
plt.plot(graphTime,graphHeight,'k', ms=4, label='Numerical Approach')
if waveType=="regular":
    plt.plot(maxInterT,d,'g.',ms=6)
    plt.axis([tBegin, tEnd, d-(2*H), d+(2*H)])

elif waveType=="irregular":
    plt.axis([tBegin, tEnd, d-(1.2*(maxHo-d)), d+(1.2*(maxHo-d))])


plt.xlabel('Time (s)')
plt.ylabel(r'$\eta$ (m)')
plt.grid(True)
pp = PdfPages('SurfEvol.pdf')
pp.savefig(fig5)
pp.close()

#======================= Frequency-Amplitude Spectrum ==========================

fig2=plt.figure(2)

if waveType=="regular":
   plt.plot(frequency,Ho,'r.', label='Analytical Solution')
   plt.bar(frq,abs(cO),width=0.002, color="black", label='Numerical Approach')

elif waveType=="irregular":
   plt.plot(frequency,Ho,'r', label='Analytical Solution')
   plt.bar(frq,abs(cO),width=0.005, color="black", label='Numerical Approach')

plt.legend(loc='upper right', shadow=True)
if waveType=="regular":
    plt.axis([0, (maxfrq+0.3*maxfrq), 0, ((H/2)+(H/4))])
elif waveType=="irregular":
    plt.axis([0, (maxfrq+0.3*maxfrq), 0, ((maxH/2)+(maxH/4))])

plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (m)')
pp = PdfPages('Spectrum.pdf')
plt.grid(True)
pp.savefig(fig2)
pp.close()
plt.show() 


# =================Definition of the analytical solution=================

if waveType=="regular":
 print(" Now you can focus on a defined interval inside the time domain and compare the analytical function with the numerical results")

 test=0

 while 1>0:
     
     response =raw_input("Do you want to compare a new time interval domain, please answer yes/no: ")
     if response=="yes":
        test=test+1
        t1 = float(raw_input("Please enter the intial time: "))
        t2 = float(raw_input("Please enter the final time: "))
 
        graphTime2=[] 
        R=[]    
         
        # Looking where the analytical function is going to start

        sequence=np.arange(0,7,1)
        q3=0
        for i in sequence:

          counter1=0
          counter2=0
          graphHeight1=[]
          graphTime1=[]
          intH1=[]
          intT1=[]
          intH2=[]
          intT2=[] 
          for index, item in enumerate(graphTime):
              if (graphTime[index]<(t1+2*T+i*T) and graphTime[index]>(t1+i*T)): 
                 graphTime1.append(graphTime[index])
                 graphHeight1.append(graphHeight[index])     
              else : continue

          for index,item in enumerate(graphHeight1):

             if graphHeight1[index]<d and graphHeight1[index+1]>d and counter1==0:

                  counter1=1
                  intH1.append(graphHeight1[index])
                  intH1.append(graphHeight1[index+1])
                  intT1.append(graphTime1[index])
                  intT1.append(graphTime1[index+1])
                  p1 = interpolate.interp1d(intH1,intT1)
                  q1=p1(d)
                  if i==0:
                     q4=q1
             if graphHeight1[index]>d and graphHeight1[index+1]<d and counter1==1 and counter2==0:

                  counter2=1
                  intH2.append(graphHeight1[index+1])
                  intH2.append(graphHeight1[index])
                  intT2.append(graphTime1[index+1])
                  intT2.append(graphTime1[index])
                  p2 = interpolate.interp1d(intH2,intT2)
                  q2=p2(d)
                  break

          q3=((q2-q1)/2.0)+q3

        q3=(q3/(len(sequence)))+q4
        graphTime2=[q3]

        for index, item in enumerate(graphTime):
           if graphTime[index]>q3:
               graphTime2.append(graphTime[index])
           else : continue
        if waveTheory=="StokesI":  
       
          # 1 Order Stokes

           def g(t):
                 return d+(H/2)*np.cos(-(2*np.pi*np.array(t)/T)+(2*np.pi*q3/T)) 
        elif waveTheory=="StokesII":

           Time=np.arange(-T/4, T/4, 0.0001*H)        
           for index, item in enumerate(Time):
                 if abs((H/2)*np.cos(-(w*Time[index])+(np.pi/2))-k*((np.power(H,2))/4)*((3-np.power(np.tanh(kd),2))/(4*(np.power(np.tanh(kd),3))))*np.cos(-2*(np.pi/2)-2*Time[index]*w))<=0.001*H: # Reduce this threshold to optimize the time, but loosing precision
                      q2=Time[index]
                 else : continue
          
          # 2 Order Stokes
          
           def g1(t):
                 return (H/2)*np.cos(-(2*np.pi*np.array(t)/T)+(2*np.pi*(q1+q2)/T)+(np.pi/2))
           def g2(t):
                 return k*((np.power(H,2))/4)*((3-np.power(np.tanh(kd),2))/(4*(np.power(np.tanh(kd),3))))*np.cos(2*(2*np.pi*(q1+q2)/T)+2*(np.pi/2)-2*np.array(t)*w)
           def g(t):
                 return d+g1(t)+g2(t) 
          
        for l in graphTime2:
           K=g(l)
           R.append(K)
        
        # Errors

        counter=0
        Dnum=[]
        Tnum=[]
        ErrorH=[]
        ErrorT=[]
        TnumMax=[]
        TnumMin=[]

        T1=[]
        T2=[]

        for index, item in enumerate(graphTime):
            if item < t2 and item >t1:
               if item >= t1+counter*T*1 and item <= t1+counter*T*1+T*1:
                  Dnum.append(graphHeight[index])         
                  Tnum.append(graphTime[index])
                  if graphTime[index+1]>t1+counter*T*1+T*1:
        
                    counter=counter+1
                    
                    DnumMax=max(Dnum)
                    DnumMin=min(Dnum)
                    TnumMax.append(Tnum[np.argmax(Dnum)])
                    TnumMin.append(Tnum[np.argmin(Dnum)])
               
                    ErrorH.append(abs((DnumMax-DnumMin)-H))

                    Dnum=[]
                    Tnum=[]

        for index,item in enumerate(TnumMax):
            
            if index+1==len(TnumMax) : break
            else:
              T1.append(TnumMax[index+1]-TnumMax[index])
        for index,item in enumerate(TnumMin):
            
            if index+1==len(TnumMin) : break
            else:
              T2.append(TnumMin[index+1]-TnumMin[index])

        T1=sum(T1)/len(T1)
        T2=sum(T2)/len(T2)
        
        ErrorT=(abs(((T1+T2)/2)-T)/T)*100     

        ErrorH=((sum(ErrorH)/len(ErrorH))/H)*100

        print('Error (Hnum-H)/H:')
        print(ErrorH) 

        print('Error (Tnum-T)/T:')
        print(ErrorT) 
                      
       
        # ______________________Plot Analytical solution___________________

        # Surface evolution in time

        fig3=plt.figure(num=3)
        plt.plot(graphTime,graphHeight,'k', ms=4, label='Numerical Approach')       
        plt.plot(graphTime2,R,'r', label='Analytical Solution')
        plt.axis([t1, t2, d-(2*H), d+(2*H)])
        plt.legend(loc='upper right', shadow=True)
        plt.xlabel('Time (s)')
        plt.ylabel(r'$\eta$ (m)')
        plt.grid(True)
        plt.show()
        pp = PdfPages('SurfEvol'+"%.0f"%test)
        pp.savefig(fig3)
        pp.close()
     else:break

file = open('SurfEvol.txt','w')
np.savetxt(file, graphHeight,fmt=['%.6f'])
file.close() 
file = open('graphTime.txt','w')
np.savetxt(file, graphTime,fmt=['%.6f'])
file.close() 
print ('x..............................................................x')
print ('x...             Post-pro Over                              ...x')
print ('x..............................................................x')
