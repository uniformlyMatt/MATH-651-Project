""" Document for edge_prog_2d.py

This module is for image edge detection with 2 dimensional wavelet transform.

If you have any problem please contact: da.li1@ucalgary.ca.
"""
import numpy as np
from scipy import signal

def gaus(N, sigma, scale):
    """ The Gaussian function.
    Input:
        N: the length of signal.
        sigma: parameter in Gaussian function.
        scale: scale used in continuous wavelet transform.
    Output:
        g: Gaussian function.
        t: time.
    """
    t = np.linspace(-5,5,N)
    g = 1/np.sqrt(scale) * 1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-1/2 * ((-1*t/scale)/sigma)**2)
    g = g / np.sqrt(np.sum(g*g))
    return g, t

def gaus1(N, sigma, scale):
    """ The first derivative of Gaussian.
    Input:
        N: the length of signal.
        sigma: parameter in Gaussian function.
        scale: scale used in continuous wavelet transform.
    Output:
        g: The first derivative of Gaussian.
        t: time.
    """
    t = np.linspace(-5,5,N)
    g = -1 * 1/np.sqrt(scale) * 1/(sigma**3*np.sqrt(2*np.pi)) * (-1*t/scale) * np.exp(-1/2 * (-1*(t/scale)/sigma)**2)
    g = g / np.sqrt(np.sum(g*g))
    return g, t

def cwt_2d(I, scale):
    """ 1 dimensional continuous wavelet transform function with first derivative of Gaussian as the wavelet.
    Input:
        I: input image, should be size N by N.
        scale: scale number in CWT.
    Output:
        H: CWT in horizontal
        V: CWT in vertical
    """
    N = I.shape[0]

    H = np.zeros([N,N])
    V = np.zeros([N,N])

    wavelet, t = gaus1(N,1,scale)

    for i in range(N):
        H[i,:] = signal.fftconvolve(I[i,:], wavelet, 'same')
    for i in range(N):
        V[:,i] = signal.fftconvolve(I[:,i], wavelet, 'same')
    return H, V

def wtmm_2d(H, V):
    """ Find the position of wavelet transform modulus maximum of an image.
    Input:
        H: CWT in horizontal
        V: CWT in vertical
    Output:
        M: wavelet modulus.
        A: Angular.
        MM: wavelet transform modulus maximum.
    """
    M = np.sqrt(H**2 + V**2)
    N = M.shape[1]

    A = V / H
    A = np.arctan(A)
    ind = np.where(H<0)

    ind1 = np.tan(22.5/180*np.pi)
    ind2 = np.tan(67.5/180*np.pi)

    MM = np.zeros([N,N])

    for i in range(1,N-2):
        for j in range(1,N-2):
            p = M[i,j]
            p1 = []; p2 = []
            if np.abs(A[i,j]) >= ind2:
                p1 = M[i+1,j]; p2 = M[i-1,j]
                if (p-p1)*(p-p2) >= 0:
                    MM[i,j] = p;
            elif np.abs(A[i,j]) <= ind1:
                p1 = M[i,j+1]; p2 = M[i,j-1]
                if (p-p1)*(p-p2) >= 0:
                    MM[i,j] = p;
            elif (A[i,j]>ind1) & (A[i,j]<ind2):
                p1 = M[i+1,j+1]; p2 = M[i-1,j-1]
                if (p-p1)*(p-p2) >= 0:
                    MM[i,j] = p;
            elif (A[i,j]>-ind2) & (A[i,j]<-ind1):
                p1 = M[i+1,j-1]; p2 = M[i-1,j+1]
                if (p-p1)*(p-p2) >= 0:
                    MM[i,j] = p;
    A[ind] = A[ind] + np.pi
    return M, A, MM

def cal_multi_edge(I, scale):
    """ Calculate for edges in all scales. This is a component of calculating Lipschitz exponent.
    Input:
        I: image.
        scale: different scales.
    Output:
        E: multiple edge file, a ndarray in size of N by N by len(scale).
    """
    E = np.ndarray([I.shape[0],I.shape[1],scale.shape[0]])
    for i in range(scale.shape[0]):
        H, V = cwt_2d(I, scale[i])
        M, A, E[:,:,i] = wtmm_2d(H, V)
    return E

def LE_2d(E, scale, th):
    """ Calculate Lipschitz exponent in 2 dimension.
    Input:
        E: multiple edge file.
        scale: different scales same as 'cal_multi_edge(I, scale):'.
        th: threshold for Lipschitz exponent.
    Output:
        alpha: Lipschitz exponent.
        ind: position of the singular point.
        val1: WTMM propagation line log_2(C) for each singular point.
        scale1: scale in log_2.
    """
    I = E[:,:,0]

    ind = np.where(np.abs(I) > th * np.max(I))
    ind_x = ind[1]
    ind_y = ind[0]

    val = np.zeros([len(scale), len(ind[0])])

    val[0,:] = I[ind]

    for i in range(1,E.shape[2]):
        I = E[:,:,i]
        for j in range(len(ind_x)):
            x_coor = ind_x[j]
            y_coor = ind_y[j]
            if x_coor>0 & y_coor>0:
                if I[x_coor, y_coor] > th*np.max(I):
                    val[i,j] = I[x_coor, y_coor]
                elif I[x_coor, y_coor-1] > th*np.max(I):
                    val[i,j] = I[x_coor, y_coor-1]
                    ind_x[j] = x_coor
                    ind_y[j] = y_coor-1
                elif I[x_coor, y_coor+1] > th*np.max(I):
                    val[i,j] = I[x_coor, y_coor+1]
                    ind_x[j] = x_coor
                    ind_y[j] = y_coor+1
                elif I[x_coor+1, y_coor] > th*np.max(I):
                    val[i,j] = I[x_coor+1, y_coor]
                    ind_x[j] = x_coor+1
                    ind_y[j] = y_coor
                elif I[x_coor+1, y_coor-1] > th*np.max(I):
                    val[i,j] = I[x_coor+1, y_coor-1]
                    ind_x[j] = x_coor+1
                    ind_y[j] = y_coor-1
                elif I[x_coor+1, y_coor+1] > th*np.max(I):
                    val[i,j] = I[x_coor+1, y_coor+1]
                    ind_x[j] = x_coor+1
                    ind_y[j] = y_coor+1
                elif I[x_coor-1, y_coor] > th*np.max(I):
                    val[i,j] = I[x_coor-1, y_coor]
                    ind_x[j] = x_coor-1
                    ind_y[j] = y_coor
                elif I[x_coor-1, y_coor-1] > th*np.max(I):
                    val[i,j] = I[x_coor-1, y_coor-1]
                    ind_x[j] = x_coor-1
                    ind_y[j] = y_coor-1
                elif I[x_coor-1, y_coor+1] > th*np.max(I):
                    val[i,j] = I[x_coor-1, y_coor+1]
                    ind_x[j] = x_coor-1
                    ind_y[j] = y_coor+1
                else:
                    ind_x[j] = -9999
                    ind_y[j] = -9999

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
        alpha[i] = m - 1

    ind = np.where(np.abs(E[:,:,0]) > th * np.max(E[:,:,0]))

    return alpha, ind, val1, scale1

def ex_edge_by_Lip(E, alpha, index, th):
    """ Extract edge file through image.
    Input:
        E: multiple edge file.
        alpha: Lipschitz exponent.
        index: position of the singular point.
        th: threshold for Lipschitz exponent.
    Output:
        Lip: image file which edge value is Lipschitz exponent.
        E_one: image file which edge value is 1.
        E_val: image file which edge value is image value itself. This is for future amplitude threshold.
    """
    II = E[:,:,0]
    A = np.zeros(II.shape)
    Lip = np.zeros(II.shape)
    E_one = np.zeros(II.shape)
    E_val = np.zeros(II.shape)
    A[index] = alpha
    index0 = np.where(A<th)
    Lip[index0] = A[index0]
    E_one[index0] = 1
    E_val[index0] = II[index0]
    return Lip, E_one, E_val
