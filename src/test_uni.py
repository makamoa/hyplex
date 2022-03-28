import numpy as np 
import matplotlib.pyplot as plt 
import math 
import sys 
import scipy.io as sio 
from numpy.linalg import solve
from scipy.linalg import expm
from numpy import linalg as LA

import tdcmt
import pso

#generate a random field
np.random.seed(14)
M=7 #n modes
N=2
w=np.linspace(0.5,4,5)
wm=w.max()*2
H=np.random.rand(N,N)
H=H+H.transpose()
C=expm(1j*H)
K=np.random.rand(M,N)+1j*np.random.rand(M,N)
om=np.diag(np.random.rand(M)*wm)
sp=np.zeros(N)
sp[0]=1
#init and plot target
td=tdcmt.tdcmt(om,C,K)
a=td.solve(w,sp)
# extract amp and phase on all channels
re,im=td.ri(a)

re_t=re[0,:]
im_t=im[0,:]


#plot it
plt.figure(1)
#plt.clf()
plt.scatter(w,re_t,s=50,c='k')

plt.figure(2)
#plt.clf()
plt.scatter(w,im_t,s=50,c='k')

plt.show()

#sys.exit(0)

# optimization function
def myfun(x):
    re,im=genfun(x)
    #calculates norm
    b=LA.norm(re-re_t)/LA.norm(re_t)+LA.norm(im-im_t)/LA.norm(im_t)
    return b

def genfun(x):
    om=np.diag(x)
    
    # init tdcmt
    td1=tdcmt.tdcmt(om,C,K)
    a=td1.solve(w,sp)
    # extract amp and phase on all channels
    re,im=td1.ri(a)
    return re[0,:],im[0,:]


def bond(x):
    for i in range(len(x)):
        if(x[i]>lb[i]):
            x[i]=np.mod(x[i],ub[i]) #modulo
            #x[i]=ub[i]
        if(x[i]<lb[i]):
            x[i]=lb[i]
            #tmp=np.abs(x[i]-lb[i])
            #x[i]=ub[i]-tmp


def init(x,v):
    tmp=np.random.rand(M)*ub[0]
    for i in range(len(x)):
        x[i]=tmp[i]
    mn=ub-lb
    tmp=(np.random.rand(M)*2.-1.)*mn
    for i in range(len(x)):
        v[i]=tmp[i]


######## search function
def search(M):

    # generate TDCMT field fixed matrices
    #M=12 #modes
    # search
    opts={
        'dim':M,
        'rng':5,
        'swarmSize':100,
        'lb':lb,
        'ub':ub,
        'maxIter':300,
        'inertiaR':np.array([0.1,1.1]),
        'minNfrac':0.5,
        'cognitive':1.49,
        'social':1.49,
        'tol':1e-5,
        'fun':myfun,
        'boundary':bond,
        'init_part':init
        }
    ps=pso.pswarm(opts)
    x0=ps.gxbest
    return x0,ps.fbest

M=12
N=2 #channels
lb=np.zeros(M)
ub=np.ones(M)*wm
H=np.random.rand(N,N)
H=H+H.transpose()
C=expm(1j*H)
K=np.random.rand(M,N)+1j*np.random.rand(M,N)

x0,fbest=search(M)

#save data
sio.savemat('pr_%i.mat' % M,{'fbest':fbest})

#sys.exit(0)

om=np.diag(x0)    
# init tdcmt
td1=tdcmt.tdcmt(om,C,K)
w1=np.linspace(w[0],w[-1],500)
a=td1.solve(w1,sp)
# extract amp and phase on all channels
re,im=td1.ri(a)
re=re[0,:]
im=im[0,:]
plt.figure(1,figsize=[2.3,1.7])
plt.plot(w1,re,'b')
plt.plot(w,re_t,'ro')
plt.ylim([-1,1])
plt.xticks([1,2,3,4])
plt.yticks([-1,0,1])
plt.tick_params(axis='both',labelleft='off', labeltop='off', labelright='off', labelbottom='off')
plt.savefig('real.pdf',dpi=300,transparent=True)
plt.close()
plt.figure(2,figsize=[2.3,1.7])
plt.plot(w1,im,'b')
plt.plot(w,im_t,'ro')
plt.ylim([-1,1])
plt.xticks([1,2,3,4])
plt.yticks([-1,0,1])
plt.tick_params(axis='both',labelleft='off', labeltop='off', labelright='off', labelbottom='off')
plt.savefig('imag.pdf',dpi=300,transparent=True)
plt.close()

#plot network
plt.figure(3,figsize=[2.3,1.7])
plt.clf()
ga=np.abs(np.diag(td1.G))
#draw connections
for i in range(M):
    for j in range(M):
        if(i!=j):
            cij=td1.G[i,j]
            plt.plot([ga[i],ga[j]],[x0[i],x0[j]],'#FBB040',linewidth=np.abs(cij)*1.25,zorder=0)
#plot nodes
plt.scatter(ga,x0,s=40,c='k',zorder=10)
plt.tick_params(axis='both',right='on',left='off',labelleft='off', labeltop='off', labelright='off', labelbottom='off')
plt.savefig('network.pdf',dpi=300,transparent=True)
plt.close()
