#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 14:44:03 2018

@author: mak
"""

import scipy.io as sio
import numpy as np
from matplotlib.pyplot import plot, xlabel, ylabel, xlim, ylim

data = sio.loadmat('ECoG-1.mat')

data.keys()

E1 = data['E1']
E2 = data['E2']
t = data['t'][0]

#%%
plt.plot(t,E1[0,:], 'b')
plt.plot(t,E2[0,:], 'r')
plt.xlabel('Time [s]');
plt.ylabel('Voltage [mV]')

#%%

ntrials = np.shape(E1)[0]
ax = plt.gca()
ax.imshow(E1, extent=[np.min(t), np.max(t), ntrials, 1])
ax.set_aspect(0.01)
plt.xlabel('Time [s]')
plt.ylabel('Trial #')
           
#%%

dt = t[1]-t[0]			#Define the sampling interval.
K = np.shape(E1)[0]			#Define the # of trials.
N = np.shape(E1)[1]         #Define number of points in each trial.
#nlags = 100			    #Define the max # of +/- lags.
ac = np.zeros([2*N-1]) #Declare empty vector for autocov.

for index,trial in enumerate(E1):				#For each trial,
    x = trial-np.mean(trial)			#...subtract the mean,
    ac0 =1/N*np.correlate(x,x,2)	#%... compute autocovar,
    ac += ac0/K;		#%...and add to total, scaled by 1/K.
    
lags = np.arange(-N+1,N)
plot(lags*dt,ac)
xlim([-0.2, 0.2])
xlabel('Lag [s]')
ylabel('Autocovariance');

#%%

x = E1[0,:] - np.mean(E1[0,:])		#Define one time series,
y = E2[0,:] - np.mean(E2[0,:])		#... and another.
xc=1/N*np.correlate(x,y,2)	#Compute trial 1 cross cov.
lags = np.arange(-N+1,N)
plot(lags*dt,xc)					#Plot cov vs lags in time.
xlim([-0.2, 0.2])
xlabel('Lag [s]')					#... with axes labelled.
ylabel('Cross covariance');

#%%
XC = np.zeros([K,2*N-1]) #Declare empty vector for cross cov.
for k in range(K):			#For each trial,
    x = E1[k,:]-np.mean(E1[k,:])			#...get data from one electrode,
    y = E2[k,:]-np.mean(E2[k,:])			#...and the other electrode,
    XC[k,:]=1/N*np.correlate(x,y,2)                    #...compute cross cov,
plt.subplot(2,1,1)
plot(lags*dt,np.mean(XC,0))					#Plot cov vs lags in time.
xlim([-0.2, 0.2])
ylim([-0.6, 0.6])
xlabel('Lag [s]')					#... with axes labelled.
ylabel('Trial-averaged cross covariance');

plt.subplot(2,1,2)
for k in range(4):
    plot(lags*dt,XC[k,:])
xlim([-0.2, 0.2])
ylim([-0.6, 0.6])
xlabel('Lag [s]')
ylabel('Cross covariance')

#%%

T = t[-1]
Sxx = np.zeros([K,int(N/2+1)])		#Create variable to store each spectrum.
for k,x in enumerate(E1):				#For each trial,
    xf  = np.fft.rfft(x-np.mean(x)) 	#... compute Fourier transform,
    Sxx[k,:] = 2*dt**2/T *np.real(xf*np.conj(xf)) #... compute spectrum.
    
f = np.fft.rfftfreq(N, dt)

plot(f,10*np.log10(np.mean(Sxx,0)))        #Plot average spectrum over trials in decibels vs frequency,
xlim([0, 100])				#... in select frequency range,
ylim([-50, 0])                  #... in select power range,
xlabel('Frequency [Hz]')	     #... with axes labelled.
ylabel('Power [ mV^2/Hz]')

plot(f,10*np.log10(Sxx[0,:]))

#%%

Sxx = np.zeros([K,int(N/2+1)])		#Create variables to save the spectra,
Syy = np.zeros([K,int(N/2+1)])
Sxy = np.zeros([K,int(N/2+1)], dtype=complex)
for k in range(K):			#For each trial,
    x=E1[k,:]-np.mean(E1[k,:])       #Get the data from each electrode,
    y=E2[k,:]-np.mean(E2[k,:])
    xf  = np.fft.rfft(x-np.mean(x)) 	#... compute Fourier transform,
    yf  = np.fft.rfft(y-np.mean(y))
    Sxx[k,:] = 2*dt**2/T *np.real(xf*np.conj(xf)) #... compute spectrum.
    Syy[k,:] = 2*dt**2/T *np.real(yf*np.conj(yf))
    Sxy[k,:] = 2*dt**2/T *       (xf*np.conj(yf))

Sxx = np.mean(Sxx,0)		#Average the spectra across trials.
Syy = np.mean(Syy,0)
Sxy = np.mean(Sxy,0)		#... and compute the coherence.

cohr = np.abs(Sxy) / (np.sqrt(Sxx) * np.sqrt(Syy))

plot(f, cohr);		#Plot coherence vs frequency,
xlim([0, 50])			#... in chosen frequency range,
ylim([0, 1])
xlabel('Frequency [Hz]')#... with axes labelled.
ylabel('Coherence')

#%%

j8 = np.where(f==8)[0][0]	 #Determine index j for frequency 8 Hz.
j24= np.where(f==24)[0][0]	#Determine index j for frequency 24 Hz.

phi8=np.zeros(K)		      #Variables to hold phase differences.
phi24=np.zeros(K)

for k in range(K):			#For each trial, compute cross spectrum, 
    x=E1[k,:]-np.mean(E1[k,:])       #Get the data from each electrode,
    y=E2[k,:]-np.mean(E2[k,:])
    xf  = np.fft.rfft(x-np.mean(x)) 	#... compute Fourier transform,
    yf  = np.fft.rfft(y-np.mean(y))
    Sxy = 2*dt**2/T *       (xf*np.conj(yf))
    phi8[k]  = np.angle(Sxy[j8])	#... and the phases.
    phi24[k] = np.angle(Sxy[j24])

plt.subplot(2,1,1)
plt.hist(phi8, bins=20, range=[-np.pi, np.pi])
ylim([0, 40])
ylabel('Counts')
plt.title('Angles at 8 Hz')
plt.subplot(2,1,2)
plt.hist(phi24, bins=20, range=[-np.pi, np.pi])
ylim([0, 40])
plt.title('Angles at 24 Hz')
ylabel('Counts')
xlabel('Phase')

#%%
N=20
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
radii, tick = np.histogram(phi24, bins = N)
width = (2*np.pi) / 20
# make a polar plot
plt.figure()
ax = plt.subplot(111, polar=True)
bars = ax.bar(theta, radii, width=width, bottom=2)
plt.show()
