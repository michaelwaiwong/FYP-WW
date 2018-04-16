#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import math
import time
layernum=3
K=3
Nt=4
Nr=2
Nj=Nr

Hidden_num_list=[[350,150]]
Data_set_list=["1400000.txt"]

N_iteration = 6000
hidden_keep_prop=1.0

#Readin data from file	
def ReadData(data_set):
	rdata=open(data_set,"r")
	tmp=rdata.read()
	Samples=tmp.split()
	InputDim=Nr*Nt*K*2
	
	OutputDim=3#2**K
	
	List1=list()
	for d in Samples:
		List1.append(float(d))
	Samples=List1
	print(len(Samples))
	N_Samples=int(len(Samples)/(InputDim+OutputDim))
	print(N_Samples,InputDim,OutputDim)
	
	Samples=np.reshape(np.matrix(Samples),(N_Samples,(InputDim+OutputDim)))
	rdata.close()
	x,y=np.split(Samples,[-(OutputDim)],axis=1)
	y=y.astype(np.int)
	return x,y,x.shape[0],InputDim,OutputDim

#Initilisation the weights and biases
def init(N_input,N_output,hidden_num):
	Weights={}
	Biases={}	
	for i in range(layernum):
		if(i==0):
			Weights['W1']=tf.Variable(tf.truncated_normal([N_input,hidden_num[i]],stddev=0.1))
			Biases['B1']=tf.Variable(tf.truncated_normal([hidden_num[i]])*0.01)
		elif(i==layernum-1):
			Weights['W'+str(layernum)]=tf.Variable(tf.truncated_normal([hidden_num[i-1],N_output],stddev=0.1))
			Biases['B'+str(layernum)]=tf.Variable(tf.truncated_normal([N_output])*0.01)
		else: 
			Weights['W'+str(i+1)]=tf.Variable(tf.truncated_normal([hidden_num[i-1],hidden_num[i]],stddev=0.1))
			Biases['B'+str(i+1)]=tf.Variable(tf.truncated_normal([hidden_num[i]])*0.01),
	return Weights, Biases
#forward
def forward(x,weights,biases):
#	x_drop=tf.nn.dropout(x,
#	x=softmax(x)
	
	z={}
	L={}	
	L[0]=x

	for i in range(layernum):
		L[i]=tf.nn.dropout(L[i],hidden_keep_prop)
		z[i+1]=tf.matmul(L[i],weights['W'+str(i+1)])+biases['B'+str(i+1)]
		if i< layernum-1:
			L[i+1]=tf.nn.relu(z[i+1])
		#relu/relu6:0.69, tanh:0.5, elu:0.619
	
	return z[layernum]

#stochastic gradient descent
def sgd(inputs,outputs,N_Samples,batch_size,initial_eta,N_input,N_output,TrainRatio,hidden_num,data_set):
	
	iterations=int(TrainRatio*N_Samples/batch_size)
	N_Train=int(iterations*batch_size)
	N_test=N_Samples-N_Train
	
	x_Train=inputs[0:N_Train,:]
	y_Train=outputs[0:N_Train,:]
	
	
	x_test=inputs[N_Train:N_Samples,:]
	y_test=outputs[N_Train:N_Samples,:]
	
	print(N_Train)
	
	X=tf.placeholder("float32", shape=[None,N_input])
	Y=tf.placeholder("float32", shape=[None,N_output])
	W,b=init(N_input,N_output,hidden_num)
	

	
	Perceptron_out=forward(X,W,b)
	
	global_step = tf.Variable(0, trainable=False)
	eta= tf.train.exponential_decay(initial_eta, global_step,10000, 1.0, staircase=True)
	entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Perceptron_out))	
	
	
	
	
	ToTrain=tf.train.AdadeltaOptimizer(learning_rate=eta).minimize(entropy)
	sess=tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	
	nn_class=tf.argmax(Perceptron_out,1)
	label_class=tf.argmax(Y,1)
	
	Right_Ans=tf.equal(nn_class,label_class)
	
	accuracy=tf.reduce_mean(tf.cast(Right_Ans,tf.float32))
	
	
	epoches_plot=[]
	Train_cross_entropy_plot=[]
	Test_cross_entropy_plot=[]
	Train_accuracy_plot=[]
	Test_accuracy_plot=[]
	
	
	t=time.time()
	for epoches in range(N_iteration):
		for Count in range(iterations):
			X_Batch=x_Train[Count*batch_size:(Count+1)*batch_size]		
			Y_Batch=y_Train[Count*batch_size:(Count+1)*batch_size]
			sess.run([ToTrain,entropy],feed_dict={X:X_Batch, Y:Y_Batch})
			
		if(epoches%3==0):
			print(epoches+1)
			print("Train_accuracy:")
			Train_accuracy=sess.run(accuracy,feed_dict={X:x_Train,Y:y_Train})	
			print(Train_accuracy)
			Test_accuracy=sess.run(accuracy,feed_dict={X:x_test,Y:y_test})
			print("Test_accuracy:")
			print(Test_accuracy)
		if(epoches%12==0):
			print("Entropy:")
			print(sess.run(entropy,feed_dict={X:x_Train,Y:y_Train}))
			print(sess.run(entropy,feed_dict={X:x_test,Y:y_test}))
			
			epoches_plot.append(epoches+1)
			Train_cross_entropy_plot.append(sess.run(entropy,feed_dict={X:x_Train,Y:y_Train}))
			Test_cross_entropy_plot.append(sess.run(entropy,feed_dict={X:x_test,Y:y_test}))
			Train_accuracy_plot.append(sess.run(accuracy,feed_dict={X:x_Train,Y:y_Train}))
			Test_accuracy_plot.append(sess.run(accuracy,feed_dict={X:x_test,Y:y_test}))
			
		if(epoches%100==0 and epoches>0):
			
			outputfilename="performance"+data_set[0:2]+"_"+str(hidden_num[0])+str(hidden_num[1])+"6000"+".txt"
			fp2=open(outputfilename,'w')
			for plotnum in range(len(epoches_plot)):
				fp2.write(str(epoches_plot[plotnum]))
				fp2.write(" ")
				fp2.write(str(Train_cross_entropy_plot[plotnum]))
				fp2.write(" ")
				fp2.write(str(Test_cross_entropy_plot[plotnum]))
				fp2.write(" ")
				fp2.write(str(Train_accuracy_plot[plotnum]))
				fp2.write(" ")
				fp2.write(str(Test_accuracy_plot[plotnum]))
				fp2.write("\n")
			fp2.close()
	
	time_elapsed=time.time()-t
	fptime=open("timetaken.txt",'a')
	fptime.write(data_set[0:7])
	fptime.write(" ")
	fptime.write(str(hidden_num[0]))
	fptime.write(" ")
	fptime.write(str(hidden_num[1]))
	fptime.write(" ")
	fptime.write(str(time_elapsed))
	fptime.write(" ")
	fptime.write('6000')
	fptime.write("\n")
	fptime.close()

	

for data_set in Data_set_list:
	x,y,N_samples,InputDim,OutputDim=ReadData(data_set)
	for hidden_num in Hidden_num_list:	
		print(x.shape)
		print(y.shape)
		sgd(x,y,int(N_samples),10000,2.0,InputDim,OutputDim,0.8,hidden_num,data_set)