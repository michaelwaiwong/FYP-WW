import numpy as np
import math
from itertools import chain, combinations
import itertools

def powerset(L,K_hat):
    pset = list()
    for n in range(2,K_hat + 1):
        for sset in itertools.combinations(L, n):
            pset.append(sset)
    return pset
tor = 0
def WaterFilling(Nj, Nr, Nt, P, N0, S, EffectiveHList):
    SizeS = len(S)
    A = np.zeros((SizeS, min(Nr, Nj)))
    SingularValueArray = np.zeros((SizeS, min(Nr, Nj)))
    for j in range(SizeS):
        s = np.linalg.svd(EffectiveHList[j], compute_uv = False)
        SingularValueArray[j] = s
        for i in range(min(Nr, Nj)):
            if s[i]> tor :
                A[j][i] = N0/(s[i]**2)
    A=np.ndarray.tolist(A.reshape(1, SizeS*(min(Nr,Nj))))[0]
    while True:
        u = (P+sum(A))/(np.count_nonzero(A))
        m = max(A)
        if u-m < tor:
            A[A.index(m)]=0
        else :
            break
    Sol = u
    OptimalPower = np.zeros((SizeS, min(Nr, Nj)))
    Capacity = 0
    for j in range(SizeS):
        for i in range(min(Nr, Nj)):
            s = SingularValueArray[j][i]
            if s > tor:
                op = max(0, Sol - N0/s**2)
                OptimalPower[j][i] = op
                Capacity += math.log(1 + op/N0*(s**2), 2)
    SelectedUser=[]
    for j in range(SizeS):
        if sum(OptimalPower[j])>tor :
            SelectedUser.append(S[j])
    return SelectedUser,Capacity

def ESearch(UserList,K_hat,Nt,Nr,Nj,N0,P):
    SubSet_User_leq_Khat = powerset(UserList,K_hat)
    maxCapacity = 0
    bestUserSet = []
    CapacityList = []
    for S in SubSet_User_leq_Khat:
        SizeS = len(S)
        for i in range(SizeS):
            H_hat = np.zeros((1, Nt))
            for j in range(SizeS):
                if(i!=j):
                    H_hat = np.concatenate((H_hat, S[j].H), axis = 0)
            S[i].H_hat = H_hat[1:]

        EffectiveHList = list()
        for i in range(SizeS):
            S[i].SetT()
            S[i].SetEffectiveH()
            EffectiveHList.append(S[i].EffectiveH)	
        TempSelectedUser,Capacity = WaterFilling(Nj, Nr, Nt, P, N0, S, EffectiveHList)
        CapacityList.append(Capacity)
        if Capacity > maxCapacity:
            bestUserSet = TempSelectedUser
            maxCapacity = Capacity
#    print('maxCapacity',maxCapacity)
#    for u in bestUserSet:
#        print(u.index)
#    print(CapacityList)
    return bestUserSet,maxCapacity
	

