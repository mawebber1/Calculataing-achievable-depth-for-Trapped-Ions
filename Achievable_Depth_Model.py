# coding: utf-8

# Calculating achievable depth and quantum volume as a function of experimental paramaters for a trapped ion design


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
font = {'size'   : 17}

plt.rc('font', **font)
get_ipython().run_line_magic('matplotlib', 'tk')
from mpl_toolkits.axes_grid.inset_locator import inset_axes


#Shuttling dependence from ion routing model. See paper for more details

#Number of shuttling operations (defined as shuttling between two adjacent X-Junctions) as a function of qubit number
def shuttlesFromQubit(qubit):
    return 1.3*(qubit**0.5)+2

#Number of times an ion on average passes through the centre of an X-Junction, as a function of qubit number
def xPassFromQubit(qubit):
    return (0.6/2**0.5)*(qubit**0.5)+2.4

#Depth overhead for swapping on a superconducting square grid using CQC's publically available tket
def scDepthOverhead(qubit):
    return max(1,((2.77*(qubit**0.5)-4.53)))


#Equations to calculate QV_native for the various architectures

#Convert a gate fidelity (%) into an error
def fid2Error(gatefid):
    return (100-gatefid)/100

#Calculate the achievable depth as a function of the total effective error and qubit number
def depth(totalError,qubit):
    return 1/(qubit*(totalError))

#Calculate trapped ion total effective error
def ionEffectiveError(qubit,gatefid,shuttleTime,coherenceTime,ionLossRate):
    #Error from decoherence 1-e^(t/c), where t includes time spent shuttling and on separating and merging.
    errorFromShuttling=1-np.exp((-(shuttleTime*shuttlesFromQubit(qubit)+2*separationMerge)*10**-6)/coherenceTime)
    #Ion Loss error
    ionLoss=xPassFromQubit(qubit)*ionLossRate
    #Assuming gate requirement is the native two qubit gate
    errorFromGates=fid2Error(gatefid)
    return errorFromShuttling+ionLoss+errorFromGates

#QV native as a function of qubit number
def QVnative(architecture,qubit,gatefid,square):
    #ion
    if architecture==1:
        totalError=ionEffectiveError(qubit,gatefid,shuttleSpeed,coherenceTime,ionLossRate)
    #all2all
    if architecture==2:
        totalError=fid2Error(gatefid)
    #Superconducting where depth overhead scales as 2.77N^0.5 -4.53 (adjust above)
    if architecture==3:
        totalError=scDepthOverhead(qubit)*fid2Error(gatefid)
    circuitDepth=depth(totalError,qubit)
    if square==1:
        QV=min(circuitDepth,qubit)**2
    else:
        QV=min(circuitDepth,qubit)
    return QV

#The peak value of QV as a function of qubit number
def maxQVnative(architecture,qubits,gatefid):
    maxQV=0
    for i in range(0,len(qubits)):
        newQV=QVnative(architecture,qubits[i],gatefid,square)
        if newQV<maxQV:
            return maxQV
        maxQV=newQV
    return maxQV

#Qubit number range investigated
qubitmin=2
qubitmax=120
qubits=[]
for i in range(qubitmin-1,int((qubitmax+2)/2)):
    qubits.append(2*i)

#Gate fidelity range investigated
gatefidmin=99
gatefidmax=99.99
gatefids=[]
dataPoints=1000
pip=(gatefidmax-gatefidmin)/dataPoints
for i in range(0,dataPoints+1):
    gatefids.append(gatefidmin+i*pip)
    
#Convert gate fidelity % to inverse error
errors=[]
for i in range(0,len(gatefids)):
    errors.append((1/((100-gatefids[i])/100)))


#Ion trapping experimental paramaters

#Coherence time (seconds)
coherenceTime=2.13
#Likely hood of loss per X-junction travel
ionLossRate=10**-5
#Time to shuttle between two adjacent X-junctions (microseconds)
shuttleSpeed=114
#Time to perform a separation or merge (microseconds)
separationMerge=80

#square QV if square==1, else plot (QV)^0.5
square=0

#Calculate and plot

ion=[]
sc=[]
all2all=[]

for i in range(0,len(gatefids)):
    ion.append(maxQVnative(1,qubits,gatefids[i]))
    
for i in range(0,len(gatefids)):
    all2all.append(maxQVnative(2,qubits,gatefids[i]))
    
for i in range(0,len(gatefids)):
    sc.append(maxQVnative(3,qubits,gatefids[i]))

    
    
fig = plt.figure(1,(1,1)) 
ax = fig.add_subplot(1, 1, 1)    
plt.plot(errors,all2all,'r-',label="Free all to all connectivity")
plt.plot(errors,ion,'g-',label="Ions with $t/c$ \u2248 5 x$10^{-5}$")
plt.plot(errors,sc,'y-',label="Superconducting square grid")
plt.legend(loc="upper left")
plt.xlabel('1/\u03B5 ')
plt.ylabel('($QV_{native})^{1/2}$')
ax.set_xscale('log')