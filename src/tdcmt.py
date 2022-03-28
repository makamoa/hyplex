import numpy as np 
import matplotlib.pyplot as plt 
import math 
import sys 
import scipy.io as sio 
from numpy.linalg import solve
from scipy.linalg import expm, norm
import torch

class tdcmt(object):
    
    # init
    def __init__(self,C,K):
        self.C=C #C unitary
        self.K=K #couplings
        self.M=K.shape[0] #number of modes
        self.N=K.shape[1] #number of channels
        #derivative
        self.G=(K @ torch.conj(K.T))*0.5


    #solve it in spectral domain w
    def solve(self,w,sp):
        dim=len(w)
        a=torch.zeros((self.M,dim),dtype=torch.cfloat)
        for i in range(dim):
            # assemble linear system
            Y=self.K @ sp
            X=1j*(torch.eye(self.M)*w[i]-self.Om)+self.G
#            tmp=np.linalg.lstsq(X,Y,rcond=-1)
#            a[:,i]=tmp[0]
            a[:,i]=torch.linalg.solve(X,Y)
        return a

    #extract output amplitude**2 and phase on the output channels
    def ap(self,a):
        dim=a.shape[1]
        amp2=np.zeros((self.N,dim))
        ph=np.zeros((self.N,dim))
        for i in range(dim):
            # calculate output            
            s=self.C[:,0]-(self.C @ (torch.conj(self.K.T) @ a[:,i]))
            #extract data
            amp2[:,i]=np.abs(s)**2
            ph[:,i]=np.angle(s)
        return amp2,ph

    def ri(self,a):
        dim=a.shape[1]
        re=torch.zeros((self.N,dim))
        im=torch.zeros((self.N,dim))
        for i in range(dim):
            # calculate output
            s=self.C[:,0]-(self.C @ (torch.conj(self.K.T) @ a[:,i]))
            #s = self.C @ np.array([1.0,0.0]) - (self.C @ (self.K.transpose() @ a[:, i]))
            #extract data
            #print('|s| =',np.abs(s), norm(s))
            re[:,i]=torch.real(s)
            im[:,i]=torch.imag(s)
        return re,im


# #test

# M=3 #modes
# N=2 #channels
# np.random.seed(M)
# om=np.diag(np.random.rand(M)*2)
# H=np.random.rand(N,N)
# H=H+H.transpose()
# C=expm(1j*H)
# K=np.random.rand(len(om),N)+1j*np.random.rand(len(om),N)
#
# #init
# td=tdcmt(om,C,K)
# #solve for one input channel exitation
# w=np.linspace(0,5,200)
# sp=np.zeros(N)
# sp[0]=1
# a=td.solve(w,sp)
# #extract amp and phase on all channels
# amp2,ph=td.ap(a)
#
# plt.figure(1)
# plt.clf()
# plt.plot(w,np.abs(a.transpose())**2)
#
# plt.figure(2)
# plt.clf()
# plt.plot(w,amp2.transpose())
#
# plt.figure(3)
# plt.clf()
# plt.plot(w,ph.transpose())
#
# plt.show()
