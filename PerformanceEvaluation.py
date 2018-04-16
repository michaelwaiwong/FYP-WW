import numpy as np
import UserGeneration as UG
import ExhaustiveSearch as ES
import math
##########################################################################
K=3
Nt=4
Nr=2
Nj=Nr
P=100
N0=1

inputDim=Nt*Nr*2*K
outputDim=K

N_traindata = 10000
K_hat = int(math.ceil(Nt/Nr))
##########################################################################
def ReadData():
    data=Datafile.readline().split()
    dataindex=1
    l=len(data)
    Re_part = []
    Im_part = []
    y_hat = []
    for d in data:
        if dataindex <= inputDim:
            if dataindex%2==1:
                Re_part.append(float(d))
            else:
                Im_part.append(float(d))
        else:
            y_hat.append(int(d)) ##
        dataindex+=1
    try:
        H = np.reshape(np.asarray(Re_part)+1j*np.asarray(Im_part),(K*Nr,Nt))
    except ValueError:
        print(l,Re_part,Im_part,y_hat)
        return [],[],0 
    Hlist = []
    for k in range(K):
        Hlist.append(H[k*Nr:(k+1)*Nr,:])
    return Hlist,y_hat,1  
##########################################################################
def ComputeCapacity(UserList):
    for i in range(K_hat):
        H_hat = np.zeros((1, Nt))
        for j in range(K_hat):
            if(i!=j):
                H_hat = np.concatenate((H_hat, UserList[j].H), axis = 0)
        UserList[i].H_hat = H_hat[1:]
    EffectiveHList = list()
    for i in range(K_hat):
        UserList[i].SetT()
        UserList[i].SetEffectiveH()
        EffectiveHList.append(UserList[i].EffectiveH)
    TempSelectedUser,Capacity = ES.WaterFilling(Nj, Nr, Nt, P, N0, UserList, EffectiveHList)
    return Capacity
############################################################################
Datafile=open("annout1800000_250250_3000.txt","r")  
#PercentCapacity=0
#MSE=0
count0=0
count01=0
count12=0
count23=0
count34=0
count45=0
count56=0
count67=0
count78=0
count89=0
count9=0
for n in range(N_traindata):
    Hlist,y_hat,check= ReadData()
    if check==0:
        print(n)
        break
    AllUserList = [None] * K
    SelectedList = []
    for k in range(K):
        AllUserList[k] = UG.User(k,Nt,Nr,Nj,2,np.asarray(Hlist[k]))
        if y_hat[k]==1:
            SelectedList.append(AllUserList[k])
    PredictedCapacity = ComputeCapacity(SelectedList)
    BestUserList, maxCapacity = ES.ESearch(AllUserList,K_hat,Nt,Nr,Nj,N0,P)

    #PercentCapacity += PredictedCapacity/maxCapacity
    #MSE+=(maxCapacity-PredictedCapacity)**2
    d = maxCapacity-PredictedCapacity
    if d <= 0.01:
        count0+=1
    if d > 0.01 and d <=1 :
        count01+=1
    if d > 1 and d <=2:
        count12+=1
    if d > 2 and d <=3:
        count23+=1
    if d > 3 and d <=4:
        count34+=1
    if d > 4 and d <=5:
        count45+=1
    if d > 5 and d <=6:
        count56+=1
    if d > 6 and d <=7:
        count67+=1
    if d > 7 and d <=8:
        count78+=1
    if d > 8 and d <=9:
        count89+=1
    if d > 9:
        count9+=1
print(count0)
print(count01)
print(count12)
print(count23)
print(count34)
print(count45)
print(count56)
print(count67)
print(count78)
print(count89)
print(count9)
 #   if n in range(100):
 #       print(PredictedCapacity, maxCapacity)
 #   if n%10000==9999:
 #       print(n+1,PercentCapacity)
#PercentCapacity/=N_traindata
#MSE/=N_traindata
#MSE = math.sqrt(MSE)
#print('average %Capacity achieved:')
#print(PercentCapacity)
#print('MSE:')
#print(MSE)
Datafile.close()
############################################################################



