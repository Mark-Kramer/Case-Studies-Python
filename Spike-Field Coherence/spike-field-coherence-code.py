# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on Wed Jun 13 09:01:28 2018

@author: mak
"""

import scipy.io as sio
import numpy as np
import statsmodels.api as sm
from scipy import signal
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.pyplot import xlabel, ylabel, plot, title
from matplotlib import rcParams
import setup
#%matplotlib inline
rcParams['figure.figsize'] = (12,3)

data = sio.loadmat('Ch11-spikes-LFP-1.mat')  # Load the multiscale data,
y = data['y']
t = data['t'].reshape(-1)
n = data['n']
plot(t,y[1,:])                               # ... and visualize it.
plot(t,n[1,:])
xlabel('Time [s]')
plt.autoscale(tight=True)                    # ... with white space minimized.

#%%

win = 100
K = np.shape(n)[0]
N = np.shape(n)[1]
STA = np.zeros([K,2*win+1])
for k in np.arange(K):
    spike_times = np.where(n[k,:]==1)
    counter=0
    for spike_t in np.nditer(spike_times):
        if win < spike_t < N-win-1:
            STA[k,:] = STA[k,:] + y[k,spike_t-win:spike_t+win+1]
            counter += 1
    STA[k,:] = STA[k,:]/counter
    
#%%
dt = t[1]-t[0]
lags = np.arange(-win,win+1)*dt
plot(lags, STA[0,:])
plot(lags, STA[5,:])
plot(lags, STA[9,:])
plot(lags, STA[15,:])
xlabel('Time [ms]')
ylabel('Voltage [mV]')

#%%
plot(lags,np.transpose(STA))
xlabel('Time [ms]')
ylabel('Voltage [mV]')

#%%

dt = t[1]-t[0]  #Define the sampling interval.
fNQ = 1/dt/2    #Define Nyquist frequency.
Wn = [44,46]  #%Set the passband
ord  = 100  #%...and filter order,
b = signal.firwin(ord, Wn, nyq=fNQ, pass_zero=False, window='hamming'); #...build bandpass filter.
FTA=np.zeros([K,N])     #Create a variable to hold the FTA.
for k in np.arange(K):    #For each trial,
    Vlo = signal.filtfilt(b, 1, y[k,:])   # ... and apply filter.
    phi = np.angle(signal.hilbert(Vlo))  # Compute phase of low-freq signal
    indices = np.argsort(phi)   #... get indices of sorted phase,
    FTA[k,:] = n[k,indices]     #... and store the sorted spikes.

#Plot the average FTA versus phase.
plot(np.linspace(-np.pi,np.pi,N), np.mean(FTA,0))

#%% Cohernece
# https://pypi.org/project/spectrum/ - does not compute coherence
# need to install this
# http://krischer.github.io/mtspec/  - doesn't look like correct n spectrum
# actually might need instead

#from spectrum import dpss, pmtm
#from mtspec import dpss, mtspec, mt_coherence
#from scipy.signal import windows   # this doesn't work.

dt = t[1]-t[0]                #Define the sampling interval.
Fs = 1/dt                         #Define the sampling frequency.
TW = 3                        #Choose time half bandwidth product of 3.
ntapers = 2*TW-1;                #Choose the # of tapers.

#test = windows.dpss(N, TW)
[tapers, lamb] = dpss(N,TW,ntapers)  # don't need this
plot(tapers)

#res = pmtm(n[3,:]-np.mean(n[k,:]), NW=10, show=False)

#%% Cohernece by hand

T = t[-1]
SYY = np.zeros(int(N/2+1))
SNN = np.zeros(int(N/2+1))
SYN = np.zeros(int(N/2+1), dtype=complex)
#S22 = np.zeros(N)

for k in np.arange(K):
    yf = np.fft.rfft((y[k,:]-np.mean(y[k,:])) *np.hanning(N))    # Hanning taper.
    nf = np.fft.rfft((n[k,:]-np.mean(n[k,:])))                   # Don't taper.
    SYY = SYY + ( np.real( yf*np.conj(yf) ) )/K
    SNN = SNN + ( np.real( nf*np.conj(nf) ) )/K
    SYN = SYN + (          yf*np.conj(nf)   )/K

cohr = np.real(SYN*np.conj(SYN)) / SYY / SNN
f = np.fft.rfftfreq(N, dt)

plt.clf()

plt.subplot(1,3,1)
plot(f,SNN)
plt.xlim([0, 100])
xlabel('Frequency [Hz]')
ylabel('Power [Hz]')
title('SNN')

plt.subplot(1,3,2)
plot(f,dt**2/T*SYY)
plt.xlim([0, 100])
xlabel('Frequency [Hz]')
ylabel('Power [Hz]')
title('SYY')

plt.subplot(1,3,3)
plot(f,cohr)
plt.xlim([0, 100])
plt.ylim([0, 1])
xlabel('Frequency [Hz]')
ylabel('Coherence')

firing_rate = np.mean(np.sum(n,1))/(N*dt)
print(firing_rate)

#%% Define a coherence function.
def coherence(n,y,t):                           #INPUT (spikes, fields, time)
    K = np.shape(n)[0]                          # spikes and fields are arrays [trials, time]
    N = np.shape(n)[1]
    T = t[-1]
    SYY = np.zeros(int(N/2+1))
    SNN = np.zeros(int(N/2+1))
    SYN = np.zeros(int(N/2+1), dtype=complex)
    
    for k in np.arange(K):
        yf = np.fft.rfft((y[k,:]-np.mean(y[k,:])) *np.hanning(N))    # Hanning taper fields
        nf = np.fft.rfft((n[k,:]-np.mean(n[k,:])))                   # Don't taper spikes.
        SYY = SYY + ( np.real( yf*np.conj(yf) ) )/K
        SNN = SNN + ( np.real( nf*np.conj(nf) ) )/K
        SYN = SYN + ( np.real( yf*np.conj(nf) ) )/K

    cohr = np.real(SYN*np.conj(SYN)) / SYY / SNN
    f = np.fft.rfftfreq(N, dt)
    
    return (cohr, f, SYY, SNN, SYN)
    

#%% scale y, no change in cohernece.
y_scaled = 0.1*y
plot(t,y[1,:])
plot(t,y_scaled[1,:])

#%%

[cohr, f, SYY, SNN, SYN] = coherence(n,y,t)
plt.clf()
plot(f,cohr)

[cohr, f, SYY, SNN, SYN] = coherence(n,y_scaled,t)
plot(f,cohr,'*')

#%% thin spikes, changes coherence.
n_thinned = np.copy(n)
thinning_factor = 0.1                #Choose a thinning factor.
for k in np.arange(K):                #For each trial,
    spike_times = np.where(n[k,:]==1)   #...find the spikes.
    n_spikes = np.size(spike_times)            #...determine # of spikes.
    spike_times_random = spike_times[0][np.random.permutation(n_spikes)]    #...permute spikes indices,
    n_remove=int(np.floor(thinning_factor*n_spikes))  # spikes to remove,
    n_thinned[k,spike_times_random[0:n_remove-1]]=0   # remove the spikes.

plt.clf()
[cohr, f, SYY, SNN, SYN] = coherence(n,y,t)
plot(f,cohr)
[cohr, f, SYY, SNN, SYN] = coherence(n_thinned,y,t)
plot(f,cohr, 'r')
plt.xlim([35, 55])

#%% Write a function to thin spike train.
def thinned_spike_train(n, thinning_factor):
    n_thinned = np.copy(n)
    for k in np.arange(K):                #For each trial,
        spike_times = np.where(n[k,:]==1)   #...find the spikes.
        n_spikes = np.size(spike_times)            #...determine # of spikes.
        spike_times_random = spike_times[0][np.random.permutation(n_spikes)]    #...permute spikes indices,
        n_remove=int(np.floor(thinning_factor*n_spikes))  # spikes to remove,
        n_thinned[k,spike_times_random[0:n_remove-1]]=0   # remove the spikes.
    return n_thinned

plt.clf()
[cohr, f, SYY, SNN, SYN] = coherence(n,y,t)
plot(f,cohr)
[cohr, f, SYY, SNN, SYN] = coherence(thinned_spike_train(n,0),y,t)
plot(f,cohr, 'r')
plt.xlim([35, 55])

#%%  GLM

dt = t[1]-t[0]  #Define the sampling interval.
fNQ = 1/dt/2    #Define Nyquist frequency.
Wn = [44,46]  #%Set the passband
ord  = 100  #%...and filter order,
b = signal.firwin(ord, Wn, nyq=fNQ, pass_zero=False, window='hamming'); #

phi=np.zeros([K,N])				#Create variable to hold phase.
for k in np.arange(K):						#For each trial,
    Vlo = signal.filtfilt(b, 1, y[k,:])   # ... and apply filter.
    phi[k,:] = np.angle(signal.hilbert(Vlo))  # Compute phase of low-freq signal

n_reshaped = np.copy(n_thinned)
n_reshaped = np.reshape(n_reshaped,-1)		# Convert spike matrix to vector.
phi        = np.reshape(phi, -1)              # Convert phase matrix to vector.
X          = np.transpose([np.ones(np.shape(phi)), np.cos(phi), np.sin(phi)])  #Create a matrix of predictors.
Y          = np.transpose([n_reshaped])                     # Create a vector of responses.

model = sm.GLM(Y,X,family=sm.families.Poisson())
res   = model.fit()

#print(res.summary())
print('Parameters: ', res.params)
print('Pvalues: ', res.pvalues)

phi_predict = np.linspace(-np.pi, np.pi, 100)
X_predict   = np.transpose([np.ones(np.shape(phi_predict)), np.cos(phi_predict), np.sin(phi_predict)])
Y_predict   = res.get_prediction(X_predict, linear='False')

Y_predict.summary_frame()

#plt.clf()
plot(np.linspace(-np.pi,np.pi,N), np.mean(FTA,0))
plot(phi_predict, Y_predict.predicted_mean, 'k')
plot(phi_predict, Y_predict.conf_int(), 'k:');

pval1=res.pvalues[1];       #Significance of parameter beta_1.
pval2=res.pvalues[2];       #Significance of parameter beta_2.

#%% GLM for null model

X0 = np.transpose([np.ones(np.shape(phi))])  #Define reduced predictor.
null_model = sm.GLM(Y,X0,family=sm.families.Poisson())  #Define reduced model
null_res   = null_model.fit()                                       #Fit reduced model.

pval = 1-stats.chi2.cdf(null_res.deviance-res.deviance,2) #Compare two nested GLMs.
print(pval)

#%% GLM with thinning factor
# Run the code above, probably should script everything.

n_thinned_reshaped   = np.reshape(thinned_spike_train(n,0.5),-1)   # Convert thinned spike matrix to vector.
Y                    = np.transpose([n_thinned_reshaped])          # Create a vector of responses.

thinned_model = sm.GLM(Y,X,family=sm.families.Poisson())           # Build the GLM model,
res_thinned   = thinned_model.fit()                                # ... and fit it.

np.exp(res_thinned.params[0])/np.exp(res.params[0])

(np.mean(np.sum(n_thinned_reshaped))/N) / (np.mean(np.sum(n_reshaped))/N)

#%%
##%% MTM Coherence by hand (doesn't work)
#
#from spectrum import dpss, pmtm
#
#T = t[-1]
#S11 = np.zeros(int(N/2+1))
##S12 = np.zeros(N)
##S22 = np.zeros(N)
#
#TW = 3
#ntapers= 2*TW-1
#[tapers, eigenvalues] = dpss(N,TW,ntapers)
#tapers = np.transpose(tapers)
#
##
##
##for k in np.arange(K):
##    yf = np.fft.rfft(y[k,:])
##    nf = np.fft.rfft(n[k,:])
##
#yf = np.zeros([int(ntapers),int(N)], dtype=complex)
#Yf = np.zeros([int(ntapers),int(N)])
#nf = np.zeros([int(ntapers),int(N)], dtype=complex)
#Nf = np.zeros([int(ntapers),int(N)])
#YNf= np.zeros([int(ntapers),int(N)], dtype=complex)
#for index, this_taper in enumerate(tapers):
#    yf[index,:] = np.fft.fft(y[k,:]*this_taper)
#    Yf[index,:] = np.real( yf[index,:] * np.conj(yf[index,:]) );
#    
#    nf[index,:] = np.fft.fft(n[k,:]*this_taper)
#    Nf[index,:] = np.real( nf[index,:] * np.conj(nf[index,:]) );
#    
#    YNf[index,:]= yf[index,:] * np.conj(nf[index,:])
#    
#SYY = np.sum(Yf,0)/K
#SNN = np.sum(Nf,0)/K
#SYN = np.sum(Yf*Nf,0, dtype=complex)/K
#cohr = np.real(SYN*np.conj(SYN)) / SYY / SNN
#
##
##res = pmtm(y[0,:], NW=3, k=ntapers, NFFT=N, show=True)
##plot(SYY)
##plot(SNN)
#plot(cohr)
