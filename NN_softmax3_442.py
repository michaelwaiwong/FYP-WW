import tensorflow as tf
import numpy as np
import math
import time
layernum = 3
K = 4
Nt = 4
Nr = 2
Nj = Nr
NewOutputDim = 6 #4C2
Hidden_num_list = [[375,150]]#,[350,200],[325,175],[300,200],[250,250]] #[L1,L2] means number of neuron in hidden layer 1 and 2.  We simulate for these 4 cases here
Data_set_list = ["aout1200000.txt","aout1400000.txt","aout1800000.txt" ] #each num.txt consists of num training data. We simulate for 1.4, 1.6, 1.8 millions here
N_iteration = 3010 #Number of iterations used for each simulation
hidden_keep_prop = 1 #each neuron has a keeping probability of 0.98 and drop out probability of 0.02

#This function is for reading data from file	
def ReadData(data_set):
	rdata = open(data_set, "r")
	tmp = rdata.read()
	Samples = tmp.split()
	InputDim = Nr*Nt*K*2
	OutputDim = NewOutputDim
	
	List1 = list()
	for d in Samples:
		List1.append(float(d))#turn the read data from string to float numbers
	Samples = List1
	print(len(Samples))
	N_Samples = int(len(Samples)/(InputDim+OutputDim))
	print(N_Samples, InputDim, OutputDim)
	Samples = np.reshape(np.matrix(Samples),(N_Samples, (InputDim+OutputDim)))
	rdata.close()
	x,y = np.split(Samples, [-(OutputDim)], axis = 1)
	y = y.astype(np.int)
	return x, y, x.shape[0], InputDim, OutputDim

#Initialize the weights and biases
def init(N_input, N_output, hidden_num):
	Weights = {}
	Biases = {}	
	for i in range(layernum):
		if(i == 0):# initialize the weights and bias in the first layer
			Weights['W1'] = tf.Variable(tf.truncated_normal([N_input, hidden_num[i]], stddev = 0.1))
			Biases['B1'] = tf.Variable(tf.truncated_normal([hidden_num[i]])*0.01)
		elif(i == layernum-1):# initialize the weights and bias in the output layer
			Weights['W'+str(layernum)] = tf.Variable(tf.truncated_normal([hidden_num[i-1], N_output], stddev = 0.1))
			Biases['B'+str(layernum)] = tf.Variable(tf.truncated_normal([N_output])*0.01)
		else: # initialize the weights and biases in the hidden layer
			Weights['W'+str(i+1)] = tf.Variable(tf.truncated_normal([hidden_num[i-1], hidden_num[i]], stddev = 0.1))
			Biases['B'+str(i+1)] = tf.Variable(tf.truncated_normal([hidden_num[i]])*0.01),
	return Weights, Biases

def forward(x, weights, biases):#pass the data x from the input layer forwardly to other layers with the given weights and biases
	z = {}
	L = {}	
	L[0] = x
	for i in range(layernum):
		L[i] = tf.nn.dropout(L[i], hidden_keep_prop)
		z[i+1] = tf.matmul(L[i], weights['W'+str(i+1)])+biases['B'+str(i+1)]
		if i < layernum-1:
			L[i+1] = tf.nn.relu(z[i+1])
		#relu/relu6:0.69, tanh:0.5, elu:0.619
	return z[layernum]

#stochastic gradient descent
def sgd(inputs, outputs, N_Samples, batch_size, initial_eta, N_input, N_output, TrainRatio, hidden_num, data_set):
	
	iterations = int(TrainRatio*N_Samples/batch_size)
	N_Train = int(iterations*batch_size)#calculate the largest size of training dataset which is divisible by the required batch_size and has a proportion not greater than the training ratio
	N_test = N_Samples-N_Train#calculate the size of the remaining data, i.e. the size of the testing dataset
	
	#split the dataset into 2 parts, 1 part is for training while the other part is for testing
	x_Train = inputs[0:N_Train, :]
	y_Train = outputs[0:N_Train, :]

	x_test = inputs[N_Train:N_Samples, :]
	y_test = outputs[N_Train:N_Samples, :]
	
	print(N_Train)
	
	X = tf.placeholder("float32", shape = [None,N_input])#a placeholder where features of each sample will be put into
	Y = tf.placeholder("float32", shape = [None,N_output])#a placeholder where label of each sample will be put into
	W, b = init(N_input, N_output, hidden_num)#initialize W, b by the function 'init'
	Perceptron_out = forward(X, W, b)#To get output from NN whenever inputs are put into the placeholder X
	global_step = tf.Variable(0, trainable = False)
	eta = tf.train.exponential_decay(initial_eta, global_step, 10000, 0.95, staircase = True)#To tune the learning rate so that the NN can converge faster.
	entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = Perceptron_out))#calculate the cost function, i.e. cross entropy function
	rgl=tf.contrib.layers.l2_regularizer(0.000005)#add a L2 regularizer with parameter 0.000005
	regularizer_W=rgl(W['W1'])
	for i in range(1,layernum):
		regularizer_W=regularizer_W+rgl(W['W'+str(i+1)])
	ToTrain = tf.train.AdadeltaOptimizer(learning_rate = eta).minimize(entropy)#Ttrain the NN with Adadelta Optimizer
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	
	nn_class = tf.argmax(Perceptron_out, 1)#classify given data
	label_class = tf.argmax(Y, 1)#get the true class of the given data
	Right_Ans = tf.equal(nn_class, label_class)#check if the two labels are equal
	accuracy = tf.reduce_mean(tf.cast(Right_Ans, tf.float32))#calculate the accuracy
	
	epoches_plot = []#initialize an array which will output performance measures of the NN for plotting graphs
	Train_cross_entropy_plot = []#calculate the cross entropy for training data
	Test_cross_entropy_plot = []#calculate the cross entropy for testing data
	Train_accuracy_plot = []#calculate the accuracy for training data
	Test_accuracy_plot = []#calculate the accuracy for testing data
	
	t = time.time()#get the current time for later testing on efficiency
	for epoches in range(N_iteration):
		for Count in range(iterations):
			X_Batch = x_Train[Count*batch_size:(Count+1)*batch_size]#get a small batch for training	
			Y_Batch = y_Train[Count*batch_size:(Count+1)*batch_size]#get the corresponding label
			sess.run([ToTrain,entropy], feed_dict = {X:X_Batch, Y:Y_Batch})#use the small batch just mentioned to train the NN
			
		if(epoches%5 == 0):#print the accuracy after every 3 iterations
			print(epoches+1)
			print("Train_accuracy:")
			Train_accuracy = sess.run(accuracy, feed_dict = {X:x_Train, Y:y_Train})	
			print(Train_accuracy)
			Test_accuracy = sess.run(accuracy, feed_dict = {X:x_test, Y:y_test})
			print("Test_accuracy:")
			print(Test_accuracy)
		if(epoches%10 == 0):#print the cross entropy and record some performance after every 12 iterations
			print("Entropy:")
			print(sess.run(entropy,feed_dict = {X:x_Train, Y:y_Train}))
			print(sess.run(entropy,feed_dict = {X:x_test, Y:y_test}))
			epoches_plot.append(epoches+1)#record some performance for plotting graphs
			Train_cross_entropy_plot.append(sess.run(entropy, feed_dict = {X:x_Train, Y:y_Train}))
			Test_cross_entropy_plot.append(sess.run(entropy, feed_dict = {X:x_test, Y:y_test}))
			Train_accuracy_plot.append(sess.run(accuracy,feed_dict = {X:x_Train, Y:y_Train}))
			Test_accuracy_plot.append(sess.run(accuracy,feed_dict = {X:x_test, Y:y_test}))

		if(epoches==500 or epoches==1000 or epoches==1500 or epoches==2000 or epoches==3000):
			outfilename = "annout"+data_set[4:-4] + "_" + str(hidden_num[0]) + str(hidden_num[1]) +"_"+str(epoches)+".txt"
			fp = open(outfilename, 'w')
			y_hat=sess.run(nn_class,feed_dict={X:x_test})
			for p in range(10000):
#				if(p%10000==9999):
#					print(p+1)
				for q in range(InputDim):
					fp.write( '%1.4f' % (x_test[p,q]))
					fp.write(" ")
				for r in range(OutputDim):
					if(r==y_hat[p]):
						fp.write(str(1))
					else:
						fp.write(str(0))
					fp.write(" ")
				if(p<N_test-1):
					fp.write("\n")
			fp.close()

		
		if(epoches%100 == 0 and epoches > 0):#save the performance measures
			outputfilename = "performance"+data_set[4:-4] + "_" + str(hidden_num[0]) + str(hidden_num[1]) +".txt"
			fp2 = open(outputfilename, 'w')
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
	
	time_elapsed = time.time()-t#to calculate the total running time
	fptime = open("timetaken.txt", 'a')#save the time taken for each NN to run
	fptime.write(data_set[0:7])
	fptime.write(" ")
	fptime.write(str(hidden_num[0]))
	fptime.write(" ")
	fptime.write(str(hidden_num[1]))
	fptime.write(" ")
	fptime.write(str(time_elapsed))
	fptime.write("\n")
	fptime.close()
	
for data_set in Data_set_list:#repeat the process for every data set mentioned at the top
	x, y, N_samples, InputDim, OutputDim = ReadData(data_set)
	for hidden_num in Hidden_num_list:	
		print(x.shape)#print out the shape of samples matrix obtained to check if there is a bug
		print(y.shape)
		sgd(x, y, int(N_samples), 10000, 2.0, InputDim, OutputDim, 0.8, hidden_num, data_set)