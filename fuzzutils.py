import numpy as np
import random
class tactics():
    def f0(self, indata:np):# roll

        if(indata.ndim==2 and indata.shape==(128,3072)):
            ins = np.reshape(indata,(128,32,32,3))
            rseed = np.random.random(1)
            if(rseed<=0.2):
                np.roll(ins,random.randint(0,10),axis=0)
            elif(rseed<=0.4):
                np.roll(ins,random.randint(0,10),axis=1)
            elif(rseed<=0.8):
                np.roll(ins,random.randint(0,10),axis=2)
            else:
                np.roll(ins,1,axis=3)
            return np.reshape(ins,(128,3072))
    def f1(self, indata:np):# loss
        if(indata.ndim==2 and indata.shape==(128,3072)):
            ins = np.reshape(indata,(128,32,32,3))
            rseed =  np.random.random(1)
            rloc = random.randint(0,10)
            fac = np.ones_like(ins,dtype=ins.dtype)

            delta = np.random.random(1)
            if(rseed<=0.33):
                fac[rloc][:][0][0]= delta
            elif(rseed<=0.67):
                fac[rloc][0][:][0]= delta
            else:
                fac[rloc][0][0][:]= delta

            ins = ins*fac
            return np.reshape(ins,(128,3072))
        
    def f2(self, indata:np):# enhance
        if(indata.ndim==2 and indata.shape==(128,3072)):
            ins = np.reshape(indata,(128,32,32,3))
            rseed =  np.random.random(1)
            rloc = random.randint(0,10)
            fac = np.ones_like(ins,dtype=ins.dtype)
            delta = np.array(random.randint(2,3)/1.9)
            if(rseed<=0.33):
                fac[rloc][:][0][0]=delta
            elif(rseed<=0.67):
                fac[rloc][0][:][0]=delta
            else:
                fac[rloc][0][0][:]=delta

            ins = ins*fac
            return np.reshape(ins,(128,3072))
    def f3(self, indata:np):# noise
        if(indata.ndim==2 and indata.shape==(128,3072)):
            ins = np.reshape(indata,(128,32,32,3))
            fac = np.random.normal(0, 0.01, size=ins.shape).astype(ins.dtype)
            ins = ins+fac
            return np.reshape(ins,(128,3072))
