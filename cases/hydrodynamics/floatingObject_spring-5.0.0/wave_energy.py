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
import matplotlib.pyplot as plt
import cmath



#======================Defining the directories==========================

pathname = os.path.abspath('.')
controlDir = os.path.join(pathname,'system/controlDict')



#================Parameters (tBegin,tEnd,tStep)=========================

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
                  tEnd=float(tEnd)
                else:continue
            if "writeInterval" in part:
                word=line.split()
                tStep=word[1]
                tStep = tStep[:-1]
                tStep=float(tStep) 



#================Postprocessing data treatment===========================

time = np.arange(tBegin,tEnd,tStep)

graphTime = []
graphsumEnergy = []
graphsumwaterEnergy = []
graphpEnergy =[]
graphEnergy = []
graphDEnergy = []
index=[]

for t in time:

    tRound  = float(math.floor(t))
    if t - tRound < tStep-0.3*tStep:
        cTime =str(int(math.floor(t)))
    else:
        cTime = str(t)
    timeSegmentDir      = cTime+'/'
    segmentFileAlpha    = timeSegmentDir+'alpha.water' 
    segmentFilemagSqrU	= timeSegmentDir+'magSqr(U)'	 

    if not os.path.isfile(segmentFileAlpha):
        print ('E... '+'no lineFile')
        sys.exit()
    alpha   	= []
    magSqrU   	= []
    watermagSqrU = []
    domainmagSqrU = []
    with open(segmentFileAlpha) as openfile:  
        alphaDict=openfile.readlines()
        n0=float(alphaDict[20]) #index of the number of alpha water value
        alphaDict = map(str.strip, alphaDict) # Remove the \n from each component of the list
        n1=int(22+n0)
        alpha=alphaDict[22:n1]
        alpha = list(map(float, alpha))

    with open(segmentFilemagSqrU) as openfile:  
        magSqrUDict=openfile.readlines()
        n0=float(magSqrUDict[20]) #index of the number of magSqrU value
        magSqrUDict = map(str.strip, magSqrUDict) # Remove the \n from each component of the list
        n1=int(22+n0)
        magSqrU=magSqrUDict[22:n1]
        magSqrU = list(map(float, magSqrU))
    watermagSqrU = magSqrU
    watermagSqrU = [ x*y for (x,y) in zip(watermagSqrU,alpha)]
    sumwatermagSqrU = sum(watermagSqrU)
    graphsumwaterEnergy.append(sumwatermagSqrU)    

    domainmagSqrU = magSqrU
    for i in range (0,len(alpha)):
	if alpha[i]>0.1:
	    	domainmagSqrU[i]=0.5*1000*0.1*0.1*0.1*domainmagSqrU[i]
	else:
	    	domainmagSqrU[i]=0.5*1.225*0.1*0.1*0.1*domainmagSqrU[i]

    summagSqrU = sum(domainmagSqrU)
    graphsumEnergy.append(summagSqrU)
    graphTime.append(t)

    #pEnergy
    #sumAlpha=sum(alpha)*0.1*0.1*0.1*5*1000*9.81
    #graphpEnergy.append(sumAlpha)


graphsumEnergy[0]=0.0
graphsumwaterEnergy[0]=0.0
graphsumwaterEnergy=[0.5*1000*0.1*0.1*0.1*k for k in graphsumwaterEnergy]
#graphEnergy=[x+y for (x,y) in zip(graphpEnergy,graphsumwaterEnergy)]
#graphDEnergy=[x+y for (x,y) in zip(graphpEnergy,graphsumEnergy)]
#minE=min(graphEnergy)
#maxE=max(graphEnergy)
#minDE=min(graphDEnergy)
#maxDE=max(graphDEnergy)
maxDE=max(graphsumEnergy)
plt.plot(graphTime,graphsumwaterEnergy,'b',label='Water Energy')
plt.plot(graphTime,graphsumEnergy,'r',label='Domain Energy')
#plt.plot(graphTime,graphEnergy,'b',label='Water Energy')
#plt.plot(graphTime,graphDEnergy,'r',label='Domain Energy')
plt.legend(loc='upper right')
#plt.axis([tBegin,tEnd,minDE-50000,maxDE+50000])
plt.axis([tBegin,tEnd,0,maxDE+100])
plt.xlabel('time (s)')
plt.ylabel('Energy')
plt.grid(True)
plt.savefig('wave_energy.png')
plt.show()
plt.close    

#file = open('Wenergy.txt','w')
#np.savetxt(file, graphEnergy,fmt=['%.6f'])
#file.close() 
#file = open('Denergy.txt','w')
#np.savetxt(file, graphDEnergy,fmt=['%.6f'])
#file.close() 
