import astropy.io.fits as pf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import FitLikelihood
import time

k450chain='kids450fiducial.fits'

t=pf.open(k450chain)
chain=t[1].data
paramnames=chain.names[2:9]

params=np.zeros((len(chain)//10,len(paramnames)))
for i in range(len(paramnames)):
    params[:,i]=chain[paramnames[i]][9::10]

mlogL=chain['mlogL'][9::10]

fll=FitLikelihood.FitLikelihood(params,mlogL,maxorder=2,norm=1)
C=np.random.rand(len(fll.multinomialpowers))
time0=time.perf_counter()
z=fll.score(C)
time1=time.perf_counter()
timeperscore=time1-time0

print('Number of multinomial terms to fit:',len(fll.multinomialpowers))
print('Number of data points including padding:',fll.Ndata+fll.Npad)
print('Each score evaluation takes %f seconds' % timeperscore)
fll.Nreport=10**np.round(np.log10(10/timeperscore),decimals=0)
print('Will report every %i function calls to score' % fll.Nreport)

C=np.zeros(len(fll.multinomialpowers))
print('Constant model gives score',fll.score(C))


print('Optimizing...')
fll.fitmodel(C)

print('Minimization succesful?',fll.fitsuccess)
print(fll.fitmessage)

Cbest=fll.bestC
np.savez('KiDS450-approxlik.npz',C=Cbest,ctr=fll.ctr,width=fll.width)

plt.plot(np.log10(1e-5+fll.y),np.log10(1e-5+fll.model(Cbest)),'k.')
plt.xlabel('Input log10(1e-5+Lik)')
plt.ylabel('Best fit log10(1E-5+Lik)')
plt.grid()
plt.tight_layout()
plt.savefig('KiDS450-approxlik.png')
plt.clf()
