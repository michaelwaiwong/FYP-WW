import tensorflow as tf
import numpy as np
import time
N_input=48
hidden_num=[350,150]
N_output=3
layernum = 3
K = 3
Nt = 4
Nr = 2
Nj = Nr
hidden_keep_prop=1
N_iteration=1000

Hidden_num_list = [[350,150]] #[L1,L2] means number of neuron in hidden layer 1 and 2.  We simulate for these 4 cases here
Data_set_list = ["aout1400000.txt" ] #each num.txt consists of num training data. We simulate for 1.4, 1.6, 1.8 millions here


def ReadData(data_set):
	rdata = open(data_set, "r")
	tmp = rdata.read()
	Samples = tmp.split()
	InputDim = Nr*Nt*K*2
	OutputDim = K
	
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




	
def testing(inputs, outputs, N_Samples, batch_size, initial_eta, N_input, N_output, TrainRatio, hidden_num, data_set):
	W={}
	b={}
	#W,b=ReadModel()
	for i in range(layernum):
		if(i == 0):# initialize the weights and bias in the first layer
			W['W1'] = tf.Variable(tf.truncated_normal([N_input, hidden_num[i]], stddev = 0.1),name='W1')
			b['B1'] = tf.Variable(tf.truncated_normal([hidden_num[i]])*0.01,name='B1')
		elif(i == layernum-1):# initialize the weights and bias in the output layer
			W['W'+str(layernum)] = tf.Variable(tf.truncated_normal([hidden_num[i-1], N_output], stddev = 0.1),name='W'+str(layernum))
			b['B'+str(layernum)] = tf.Variable(tf.truncated_normal([N_output])*0.01,name='B'+str(layernum))
		else: # initialize the weights and biases in the hidden layer
			W['W'+str(i+1)] = tf.Variable(tf.truncated_normal([hidden_num[i-1], hidden_num[i]], stddev = 0.1),name='W'+str(i+1))
			b['B'+str(i+1)] = tf.Variable(tf.truncated_normal([hidden_num[i]])*0.01,name='B'+str(i+1)),
	N_Train = int(0.8*N_Samples)
	N_test = N_Samples-N_Train#calculate the size of the remaining data, i.e. the size of the testing dataset
	#split the dataset into 2 parts, 1 part is for training while the other part is for testing

	x_Train = inputs[0:N_Train, :]
	y_Train = outputs[0:N_Train, :]
	x_test = inputs[N_Train:N_Samples, :]
	y_test = outputs[N_Train:N_Samples, :]		
	save_path = "my_net18_350150/save_net.ckpt"
	saver=tf.train.Saver()
	init = tf.global_variables_initializer()
	sess=tf.Session()
	
	# Initialize variables
	sess.run(init)
	# Restore model weights from previously saved model
	saver.restore(sess, save_path)
	print("Model restored from file: %s" % save_path)
	#print(sess.run(W))
	#print(sess.run(b))
	W1=np.matrix(sess.run(W['W1']))
	W2=np.matrix(sess.run(W['W2']))
	W3=np.matrix(sess.run(W['W3']))
	B1=np.matrix(sess.run(b['B1']))
	B2=np.matrix(sess.run(b['B2']))
	B3=np.matrix(sess.run(b['B3']))
	#z1=sess.run(tf.nn.relu6(np.dot(x_Train,W1)+B1))
	#z2=sess.run(tf.nn.relu6(np.dot(x_Train
	#W={'W1':W1,'W2':W2,'W3':W3}
	#b={'B1':B1,'W2':B2,'B3':B3}
	#Perceptron_out = forward(x_test, W, b)
	t=time.time()
	Perceptron_out1=sess.run(tf.nn.relu(np.dot(x_test,W1)+B1))
	Perceptron_out2=sess.run(tf.nn.relu((np.dot(Perceptron_out1,W2)+B2)))
	Perceptron_out=sess.run(tf.nn.relu((np.dot(Perceptron_out2,W3)+B3)))
	x_test = inputs[N_Train:N_Samples, :]
	print(N_test)
	nn_class = sess.run(tf.argmax(Perceptron_out, 1))#classify given data
	y_head={}
	
	for i in range(N_test):
		if(nn_class[i]==0):
			y_head[i]=[0, 1, 1]
		elif(nn_class[i]==1):
			y_head[i]=[1, 0, 1]
		elif(nn_class[i]==2):
			y_head[i]=[1, 1, 0]
		else:
			print("Error")
	print(time.time()-t)




for data_set in Data_set_list:#repeat the process for every data set mentioned at the top
	x, y, N_samples, InputDim, OutputDim = ReadData(data_set)
	for hidden_num in Hidden_num_list:	
		print(x.shape)#print out the shape of samples matrix obtained to check if there is a bug
		print(y.shape)
	testing(x, y, int(N_samples), 10000, 2.0, InputDim, OutputDim, 0.8, hidden_num, data_set)
	