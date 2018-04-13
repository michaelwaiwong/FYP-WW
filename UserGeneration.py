import numpy as np
import math
from numpy import linalg as LA
from rank_nullspace import nullspace,rank
class User:
    def __init__(self, index, Nt, Nr, Nj, mode, H):
        self.index = index # the indices are used to identify different user
        self.state = 0 #1 means selected (initialized to 0)
        self.Nj = Nj
        self.Nr = Nr
        self.Nt = Nt
        if mode == 1: 
# mode==1 means to generate user with channel matrix according to i.i.d. complex Gaussian(0,1) 
            self.H = np.random.randn(self.Nr, self.Nt)
            self.H += 1j*np.random.randn(self.Nr, self.Nt)
            self.H /= math.sqrt(2)
        else:         
# else means to generate user with given channel matrix H
            self.H = H
        self.H_hat = [None]
        self.PrecodingT = [None]
        self.EffectiveH = [None]

    def SetT(self):
        if self.H_hat.shape[0] != 0 :
            NS = nullspace(self.H_hat)
            if rank(NS) >= self.Nj :
                self.PrecodingT = NS[:, 0:self.Nj]
            else:
                self.PrecodingT = [None]
        else :
            self.PrecodingT = [None]

    def SetEffectiveH(self):
        if self.H_hat.shape[0] != 0 :
            self.EffectiveH = np.dot(self.H, self.PrecodingT)
        else:
            self.EffectiveH = [None]