import numpy as np
K = 4 #3
Nt = 4
Nr = 2
Nj = Nr

rdata = open("atrain.txt","r")
wdata = open("aout.txt","w")
tmp = rdata.read()
Samples = tmp.split()
InputDim = Nr*Nt*K*2
OutputDim = K
NewOutputDim = 6 #4C2
List1 = list()
for d in Samples:
    tmp = d.replace('(', '')
    tmp = tmp.replace(')', '')
    tmp = tmp.replace('+', ' ')
    tmp = tmp.replace('j', '')
    for e in tmp.split():
        List1.append(float(e))
Samples = List1
print(len(Samples))
NSamples = int(len(Samples)/(InputDim+OutputDim))
print(NSamples)
rdata.close()
for i in range(0, NSamples):
    for j in range(0, InputDim):
        wdata.write(str(Samples[i*(InputDim+OutputDim)+j]))
        wdata.write(" ")
    tmp = i*(InputDim+OutputDim)+InputDim
    index = 0
    for k in range(0, OutputDim):           #For 4 users
        index+=Samples[tmp+k]*2**(OutputDim-k-1)
    index = int(index)
    if(index == 3):                         
        wdata.write("1 0 0 0 0 0 ")
    elif(index == 5):
        wdata.write("0 1 0 0 0 0 ")
    elif(index == 6):
        wdata.write("0 0 1 0 0 0 ")
    elif(index == 9):
        wdata.write("0 0 0 1 0 0 ")
    elif(index == 10):
        wdata.write("0 0 0 0 1 0 ")
    else: #index == 12
        wdata.write("0 0 0 0 0 1 ")

    
#    for k in range(3):                     #For 3 users
#       if(Samples[tmp+k] == 0):
 #          wdata.write("1 ")
 #      elif(Samples[tmp+k] == 1):
 #          wdata.write("0 ")
 #      else:
 #          print("Error")
    
    wdata.write("\n")
    
wdata.close()

