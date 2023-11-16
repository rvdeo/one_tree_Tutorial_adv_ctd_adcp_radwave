
import os
import math
import numpy as np
import pandas as pd
import matplotlib.mlab as mlab
from scipy.signal import hann
from pyfftw.interfaces import scipy_fftpack as fftw

def PT_power(x, dt, h=0):
    
    n = len(x)
    xd = mlab.detrend(x)
    
    if h > 0:
        xd = xd*hann(n+2)[1:-1]

    xn = fftw.fft(xd,n)
    if h > 0:
        xn = xn*np.sqrt(8/3)
        
    pxx = dt*(xn*np.conj(xn))/n
    
    paa = np.sqrt(xn*np.conj(xn))/n
    pxx = dt*(xn*np.conj(xn))/n
    
    nf= int(np.fix((n/2)+1))
    px=2.0 * pxx[0:nf]
    pa=2.0 * paa[0:nf]
    step = 1/(n*dt)
    f = np.arange(0, 1/(2.*dt)+step, step)[0:-1]
    
    return px, f, pa 

def Wwaves_spectral_RBR(px_in,f_in,dt):
    
    n21 = len(px_in)
    n = (n21-1)*2
    
    jj = np.argwhere((f_in>=0.33)&(f_in<1))[:,0]
    
    f = f_in[jj]
    px = px_in[jj]
    
    m0 = np.sum(px)/(n*dt)
    m1 = np.sum(f*px)/(n*dt);
    m2 = np.sum((f**2)*px)/(n*dt)
    m4 = np.sum((f**4)*px)/(n*dt)
    
    hs = 4.*np.sqrt(m0)
    hrms = np.sqrt(8.*m0)
    tm02 = np.sqrt(m0/m2)
    tm01 = m0/m1
    eps = np.sqrt(1.-(m2**2)/(m0*m4))
    hmax = 2.*hs
    
    ii = np.argwhere(f>=0.33)[:,0]
    fc = f[ii]
    pxc = px[ii]
    j = np.abs(pxc).argmax()
    tpeak = 1/fc[j]
    
    return hs, hrms, hmax, tm01, tm02, tpeak, eps


def Gwaves_spectral_RBR(px_in,f_in,dt):
    
    n21 = len(px_in)
    n = (n21-1)*2
    
    jj = np.argwhere((f_in>=0.05)&(f_in<0.33))[:,0]
    
    f = f_in[jj]
    px = px_in[jj]
    
    m0 = np.sum(px)/(n*dt)
    m1 = np.sum(f*px)/(n*dt);
    m2 = np.sum((f**2)*px)/(n*dt)
    m4 = np.sum((f**4)*px)/(n*dt)
    
    hs = 4.*np.sqrt(m0)
    hrms = np.sqrt(8.*m0)
    tm02 = np.sqrt(m0/m2)
    tm01 = m0/m1
    eps = np.sqrt(1.-(m2**2)/(m0*m4))
    hmax = 2.*hs
    
    ii = np.argwhere(f>=0.05)[:,0]
    fc = f[ii]
    pxc = px[ii]
    j = np.abs(pxc).argmax()
    tpeak = 1/fc[j]
    
    return hs, hrms, hmax, tm01, tm02, tpeak, eps 


def IGwaves_spectral_RBR(px_in,f_in,dt):
    
    n21 = len(px_in)
    n = (n21-1)*2
    
    jj = np.argwhere((f_in>=0.0033)&(f_in<0.05))[:,0]
    
    f = f_in[jj]
    px = px_in[jj]
    
    m0 = np.sum(px)/(n*dt)
    m1 = np.sum(f*px)/(n*dt);
    m2 = np.sum((f**2)*px)/(n*dt)
    m4 = np.sum((f**4)*px)/(n*dt)
    
    hs = 4.*np.sqrt(m0)
    hrms = np.sqrt(8.*m0)
    tm02 = np.sqrt(m0/m2)
    tm01 = m0/m1
    eps = np.sqrt(1.-(m2**2)/(m0*m4))
    hmax = 2.*hs
    
    ii = np.argwhere(f>=0.0033)[:,0]
    fc = f[ii]
    pxc = px[ii]
    j = np.abs(pxc).argmax()
    tpeak = 1/fc[j]
    
    return hs, hrms, hmax, tm01, tm02, tpeak, eps 
