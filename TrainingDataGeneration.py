import numpy as np
import math
import UserGeneration as UG 
import ExhaustiveSearch as ES
K = 4  #number of user
Nt = 4  #number of transmit antennas at base station
Nr = 2  #number of receive antennas for all users
Nj = 2  #number of data streams for all users 
N0 = 1  #noise power/variance
P = 100   #total transmit power constraint of all data streams

K_hat = int(math.ceil(Nt/Nr))

if K_hat * Nr > Nt :
    print("Block Diagonalization is not possible")
    exit(1)

outfile = open('atrain.txt','a')
for ii in range(1800000):


    UserList = [None] * K
    for UserIndex in range(K) :
	    UserList[UserIndex] = UG.User(UserIndex, Nt, Nr, Nj,1,0)
    SelectedUserList, maxCapacity = ES.ESearch(UserList,K_hat,Nt,Nr,Nj,N0,P)

    for u in SelectedUserList:
	    u.state = 1

    inputChannelMatrix = np.ravel(UserList[0].H)
    outputVector = [UserList[0].state]

    for j in range(1,K):
	    inputChannelMatrix = np.concatenate((inputChannelMatrix,np.ravel(UserList[j].H)),axis = 0)
	    outputVector.append(UserList[j].state)

#print(inputChannelMatrix)
#for i in xrange(K):
#    print(vars(UserList[i]))
#print(K)
#print(outputVector)

#print(maxCapacity)

    np.savetxt(outfile,inputChannelMatrix,fmt='%1.4f',newline=' ')
    np.savetxt(outfile,[[]],newline='\n')
    np.savetxt(outfile,outputVector,fmt='%1d',newline=' ')
    np.savetxt(outfile,[[]],newline='\n')    
#        np.savetxt(outfile,[maxCapacity],fmt='%f',newline=' ')
#        np.savetxt(outfile,[[]],newline='\n')
    if ii%1000==999:
        print(ii+1)
outfile.close()

