import numpy as np
from scipy import signal

def gaus(N, sigma, scale):
    t = np.linspace(-5,5,N)
    g = 1/np.sqrt(scale) * 1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-1/2 * ((-1*t/scale)/sigma)**2)
    g = g / np.sqrt(np.sum(g*g))
    return g, t

def gaus1(N, sigma, scale):
    t = np.linspace(-5,5,N)
    g = -1 * 1/np.sqrt(scale) * 1/(sigma**3*np.sqrt(2*np.pi)) * (-1*t/scale) * np.exp(-1/2 * (-1*(t/scale)/sigma)**2)
    g = g / np.sqrt(np.sum(g*g))
    return g, t

def cwt_1d(sig, scale):
    C = np.zeros([len(scale), len(sig)])
    for i in range(len(scale)):
        wavelet, t = gaus1(len(sig),2,scale[i])
        C[i,:] = signal.fftconvolve(sig, wavelet, 'same')
    return C

def wtmm_1d(C):
    nrow = C.shape[0]; ncol = C.shape[1]
    MM = np.zeros([nrow, ncol])
    for i in range(nrow):
        left = np.zeros(ncol)
        right = np.zeros(ncol)
        left[0:ncol-2] = C[i,1:ncol-1]
        right[1:ncol-1] = C[i,0:ncol-2]
        ind = np.where((C[i,:]-left) * (C[i,:]-right) >= 0)
        MM[i,ind] = np.abs(C[i,ind])
    return MM

def LE_1d(C, scale):
    th = 0.01

    ind = np.where(np.abs(C[0,:]) > th * np.max(C))[0]
    val = np.zeros([C.shape[0], ind.shape[0]])
    val[0,:] = C[0,ind]

#     for i in range(1,C.shape[0]):
#         for j in range(ind.shape[0]):
#             ind0 = ind[j]
#             for k in range(-1*gap,gap+1):
#                 if C[i,ind0+k] > th*np.max(C):
#                     val[i,j] = C[i,ind0+k]
#                     ind[j] = ind0+k
    for i in range(1,C.shape[0]):
        for j in range(ind.shape[0]):
            ind0 = ind[j]
            if ind0 > 0:
                if C[i,ind0] > th*np.max(C):
                    val[i,j] = C[i,ind0]
                elif C[i,ind0-1] > th*np.max(C):
                    val[i,j] = C[i,ind0-1]
                    ind[j] = ind0-1
                elif C[i,ind0+1] > th*np.max(C):
                    val[i,j] = C[i,ind0+1]
                    ind[j] = ind0+1
                else:
                    ind[j] = -99999

    val1 = np.zeros(val.shape)
    for i in range(val.shape[0]):
        val1[i,:] = np.log2(val[i,:])

    alpha = np.zeros(val1.shape[1])
    scale1 = np.log2(scale)
    for i in range(val1.shape[1]):
        y = val1[:,i]
        drop_inf = np.isinf(y)
        y = y[~drop_inf]
        x = scale1[~drop_inf]
        A = np.ones([y.shape[0],2])
        A[:,0] = x
        m, c = np.linalg.lstsq(A,y)[0]
        alpha[i] = m - 1/2

    ind = np.where(np.abs(C[0,:]) > th * np.max(C))[0]

    return alpha, ind, val1, scale1
