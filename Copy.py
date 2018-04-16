K=4
Nt=4
Nr=2
Nj=Nr

InputDim=Nr*Nt*K*2
NewOutputDim = 6 #4C2
OutputDim=NewOutputDim
	
def WriteData(samples,dsize):
    outname = "aout"+str(dsize)+".txt"
    wdata=open(outname,"w")
    for i in range(dsize):
        for j in range(0,InputDim):
            wdata.write(str(samples[i*(InputDim+OutputDim)+j]))
            wdata.write(" ")
        tmp=i*(InputDim+OutputDim)+InputDim
        for k in range(0,OutputDim):
            if (samples[tmp+k]==0):
                wdata.write("0")
            elif(samples[tmp+k]==1):
                wdata.write("1")
            else: 
                print("Error")
            wdata.write(" ")
        wdata.write("\n")		
    wdata.close()
    print('finish writing outfile')

Data_set=[1200000,1400000]
rdata=open("aout1800000.txt","r")
tmp=rdata.read()
Samples=tmp.split()
List1=list()
for d in Samples:
    List1.append(float(d))
Samples=List1
rdata.close()
print('finish reading data')
for Dsize in Data_set:
    WriteData(Samples,Dsize)


