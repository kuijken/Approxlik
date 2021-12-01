import numpy as np
from scipy.optimize import minimize

class FitLikelihood:

    def __init__(self,params,minusloglik,paramranges=None,pad=True,maxorder=4,norm=1):
        ''' Read in the data:
             params=NdataxNpar parameter values
             minusloglik=corresponding Ndata -ln(lik) values
             paramranges=Npar x 2 array of parameter min, max for normalisation
               (None means min and max of supplied parameters will be used)
               [min,max] interval is remapped to [-1,1]  
             pad=True adds zero-data points all around the data
             maxorder is the highest order of multinomial terms to include
        '''
        Ndata,Npar=params.shape
        # first center on peak and normalize the data by width
        ipk=np.argmin(minusloglik)
        bestloglik=minusloglik[ipk]
        ctr=params[ipk,:]
        width=np.max(np.abs(params-ctr),axis=0)
        self.Ndata,self.Npar=Ndata,Npar
        self.x=(params-ctr)/width
        self.y=np.exp(bestloglik-minusloglik)
        self.ctr=ctr
        self.width=width
        self.pk=bestloglik
        self.Ndata,self.Npar=Ndata,Npar
        if paramranges is None:
            self.xlo=np.min(self.x,axis=0)
            self.xhi=np.max(self.x,axis=0)
        else:
            self.xlo=(paramranges[:,0]-ctr)/width
            self.xhi=(paramranges[:,1]-ctr)/width
        if pad:
            self.padzero()
        else:
            self.Npad=0
        self.multinomialorder=maxorder
        self.multinomialpowers=multpowers(Npar,maxorder)
        p=self.multinomialpowers
        self.multinomialterms=np.array([(xx**p).prod(axis=1) for xx in self.x])
        self.count=0
        self.Nreport=20000
        self.lowestscore=np.inf
        self.scorenorm=norm

    def padzero(self):
        xm=2*self.xlo-self.xhi
        xp=2*self.xhi-self.xlo
        Npad=3**self.Npar -1
        idx=np.arange(1,Npad+1)
        xpad=np.zeros((Npad,self.Npar))
        for i in range(self.Npar):
            j=idx//3**i % 3
            xpad[j==1,i]=xm[i]
            xpad[j==2,i]=xp[i]
        self.x=np.concatenate((self.x,xpad))
        self.y=np.concatenate((self.y,np.zeros(Npad)))
        self.Npad=Npad
        

    def model(self,C):
        ''' model for -ln(lik) with multinomial coeffs C
               C should have dimension (Npar+1)(Npar+2)(Npar+3)(Npar+4)/24
            return model evaluated at every data point
            write out a line every Nreport calls
        '''
        ymod=np.exp(-self.multinomialterms @ C)
        self.count+=1
        if (self.count % self.Nreport == 0):
            print('Function called %i times. Lowest score so far %g' \
                      % (self.count,self.lowestscore))
        return ymod

    def score(self,C):
        ''' average of |model-loglik| over all data '''
        s=np.mean(np.abs(self.model(C)-self.y)**self.scorenorm)
        self.lowestscore=min(self.lowestscore,s)
        return s

    def fitmodel(self,C0):
        ''' minimize the average MAD between model and -logLik'''
        minimization=minimize(self.score,C0) # run minimizer, starting with C0
        self.bestC=minimization.x
        self.nit=minimization.nit
        self.nfev=minimization.nfev
        self.fun=minimization.fun
        self.fitsuccess=minimization.success
        self.fitmessage=minimization.message
 
        print('Iterations:',minimization.nit)
        print('Function calls:',minimization.nfev)
        print('Best score:',minimization.fun)
        print('Multinomial coefficients:')
        for i in range(len(self.bestC)):
            print(self.multinomialpowers[i],'%12.5f' % self.bestC[i])
        return

    def emulate(self,pars):
        ''' return the fitted -logL evaluated at parameters array pars
             pars must have last dimension self.Npar
        '''
        # remap pars to x using width and ctr
        # calculate the monomials and reconstruct y
        # add the peak 
        ###TBD

    def loadfit(self,filename='approxlik.npz'):
        try:
            f=np.load(filename)
            self.bestC=f['C']
            self.ctr=f['ctr']
            self.width=f['width']
            self.Npar=len(self.bestC)
            print('Read fit coefficients from',filename)
            return True
        except:
            print('Cannot read fit coefficients from',filename)
            return False

    def savefit(self,filename='approxlik.npz'):
        np.savez(filename,C=self.bestC,ctr=self.ctr,width=self.width)
        print('Fit coefficients written to',filename)
    
   
def multpowers(npar,maxdim=4):
        '''
        return a list of all possible ordered sets of powers
        for npar params that sum to <=maxdim
        '''
        # think of a row of npar dividers:
        #  you can place maxdim balls in this row between the dividers
        #  the power of parameter d is the number of balls between
        #      dividers d-1 and d
        #  all balls to the right of the last divider are powers of 1
        #      (i.e., this gives a lower-degree term)
        
        #  first choose the locations of d dividers between parameters
        #  they can be in positions 0..maxdim
        #  each possibility is an element of list s:
        #      a set of d ordered, possibly equal positions
        if npar==0:
            s=[[]]   # if npar=0 there is only the empty set,
                     # regardless of order maxdim
        else:
            s=[[x] for x in range(maxdim+1)]   # for d=1, divider can be
                                               # anywhere from 0 and maxdim
            for dd in range(1,npar):   # for higher d, only add dividers on
                                       # the right to avoid double counting
                s=[x+[i] for x in s for i in range(max(x),maxdim+1)]
        # the power of parameter d is then the space between divider d and d-1
        #   (with implicitly assumed x[-1]=0)
        powers=np.array([ [x[0]]+[x[i+1]-x[i]  for i in range(npar-1)] \
                          for x in s],  dtype=int)
        return powers

    
