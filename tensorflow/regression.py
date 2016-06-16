import readData
import matplotlib.pyplot as plt
import numpy as np

from random import randint
import random
from root_numpy import fill_hist

from ROOT import TCanvas, TH2F, TText, TF1 ,TH1D
import ROOT

import tensorflow as tf

import math



# # # # # # ##
#Read in data#
#            #

function_outputs=True# apply an invertable function to the y's and train with the modified outputs y_mod! Up to know this function is just a normalization.
function_inputs=True # 


full_set =  readData.read_data_set("./TH2D_A00_TB10.root","LHCChi2_CMSSM_nObs1061_A00_TB10","full_set",function_inputs,function_outputs)

N_full_set=full_set.get_N()
N_validation_set=10000
N_training_set=N_full_set-(N_validation_set)

full=range(0,N_full_set)

random.shuffle(full)
training_subset=full[:N_training_set]#indices for training set
validation_subset=full[N_training_set:N_training_set+N_validation_set]#indices for validation set



training_set = readData.read_data_set("./TH2D_A00_TB10.root","LHCChi2_CMSSM_nObs1061_A00_TB10","training_set",
                                      function_inputs,function_outputs,full_set=full_set,subset=training_subset)
validation_set = readData.read_data_set("./TH2D_A00_TB10.root","LHCChi2_CMSSM_nObs1061_A00_TB10","validation_set",
                                   function_inputs,function_outputs,full_set=full_set,subset=validation_subset )

#overwiew of full data set, training_data set and validation_data set. The modified members( normalized in this case) can be accessed with the x_mod() and y_mod() member functions
#the normalized data (input and output) will be used to train the net
print "full_data_set:"
print "x (inputs)"
print full_set.get_x()
print "y (outputs)"
print full_set.get_y()
print "x_mod"
print full_set.get_x_mod()
print "y_mod"
print full_set.get_y_mod()
print "------------------"
print "training_data_set:"
print "x (inputs)"
print training_set.get_x()
print "y (outputs)"
print training_set.get_y()
print "x_mod"
print training_set.get_x_mod()
print "y_mod"
print training_set.get_y_mod()
print "------------------"
print "evaluation_data_set:"
print "x (inputs)"
print validation_set.get_x()
print "y (outputs)"
print validation_set.get_y()
print "x_mod"
print validation_set.get_x_mod()
print "y_mod"
print validation_set.get_y_mod()
print "------------------"





# # # # # # # # # # # ## 
#setting up the network#
#                      #
 
N_epochs = 20

learning_rate = 3.0	
batch_size = 10


N1 = 2 #equals N_inputs
N2 = 30
N3 = 30
N4 = 30
N5 = 1

N_in=N1
N_out=N5


#one calculates everything directly for all elements in one batch
"""example: N_in=2,N_out=3, mini_batch_size=5, activation function=linear. In der output matrix gibt es 5Zeilen,jede fuer ein mini batch. Jede Zeile hat 3 Spalten fuer ein output neuron jeweils

W2
[[-0.31917086 -0.03908769  0.5792625 ]
 [ 1.34563279  0.03904691  0.39674851]]
b2
[ 0.40960133 -0.5495823  -0.97048181]
x_in
[[  23.2       12.2    ]
 [   0.         1.1    ]
 [   2.3        3.3    ]
 [  23.22222   24.44444]
 [ 333.       444.     ]]
y=x_in*W2+b2
[[   9.42155647   -0.98004436   17.30874062]
 [   1.88979745   -0.50663072   -0.53405845]
 [   4.1160965    -0.51062918    1.67109203]
 [  25.8909874    -0.50280523   22.17957497]
 [ 491.5866394     3.77104688  368.08026123]]

hier wird klar, dass b2 auf jede Zeile der Matrix x_in*w2 draufaddiert wird.
W2 ist die transponierte der atrix, die im Buch definiert ist.
"""

x = tf.placeholder(tf.float32,[None,N1])#don't take the shape=(batch_size,N1) argument, because we need this for different batch sizes


W2 = tf.Variable(tf.random_normal([N1, N2],mean=0.0,stddev=1.0/math.sqrt(N1*1.0)))# Initialize the weights for one neuron with 1/sqrt(Number of weights which enter the neuron/ Number of neurons in layer before)
b2 = tf.Variable(tf.random_normal([N2]))
a2 = tf.sigmoid(tf.matmul(x, W2) + b2) #x=a1

W3 = tf.Variable(tf.random_normal([N2, N3],mean=0.0,stddev=1.0/math.sqrt(N2*1.0)))
b3 = tf.Variable(tf.random_normal([N3]))
a3 = tf.sigmoid(tf.matmul(a2, W3) + b3)

W4 = tf.Variable(tf.random_normal([N3, N4],mean=0.0,stddev=1.0/math.sqrt(N3*1.0)))
b4 = tf.Variable(tf.random_normal([N4]))
a4 = tf.sigmoid(tf.matmul(a3, W4) + b4)

W5 = tf.Variable(tf.random_normal([N4, N5],mean=0.0,stddev=1.0/math.sqrt(N4*1.0)))
b5 = tf.Variable(tf.random_normal([N5]))
y = tf.sigmoid(tf.matmul(a4, W5) + b5)

y_ = tf.placeholder(tf.float32,[None,N_out]) #  ,shape=(None,N_out)






# # # # # # # # # # # # # #
#initializing and training#
#                         #

cost_function = tf.scalar_mul(1.0/(N_training_set*2.0),tf.reduce_sum(tf.squared_difference(y,y_))) 
error_to_desired_output= y-y_
abs_error_to_desired_output= tf.abs(y-y_)
sum_abs_error_to_desired_output= tf.reduce_sum(tf.abs(y-y_))

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function) 
init = tf.initialize_all_variables()

#launch the graph
sess = tf.Session()
sess.run(init)


N_training_batch=training_set.get_N()/batch_size #rounds to samllest integer

out_mod_validation=[0]*N_epochs # output of net, when inputting x_mod of validation data. Will be saved after each epoch.
error_mod_validation_data= [0]*N_epochs #absolute error on mod validation data after each epoch
diff_mod_validation=[0]*N_epochs # error vector of validation data after each epoch. i.e. y-y_

cost_training_data=[0]*N_epochs


for i in range(0,N_epochs):
	for j in range(0,N_training_batch):
		batch_xs, batch_ys, epochs_completed = training_set.next_batch(batch_size)#always gives the modified x's and y's. If one does not want to modifie them the function has to be set to identity 
		sess.run(train_step, feed_dict={x: batch_xs, 
			y_: batch_ys})
	
	cost_training_data[i]=sess.run(cost_function, feed_dict={
		x: training_set.get_x_mod(), y_: training_set.get_y_mod()})
	out_mod_validation[i]= sess.run(y, feed_dict={
		x: validation_set.get_x_mod()})# output of net, when imputting x_mod of validation data after each training epoch
	diff_mod_validation[i]=sess.run(error_to_desired_output, feed_dict={
		x: validation_set.get_x_mod(),y_: validation_set.get_y_mod()})
	error_mod_validation_data[i]=sess.run(sum_abs_error_to_desired_output, feed_dict={
		x: validation_set.get_x_mod(),y_: validation_set.get_y_mod()})
	print "epochs completed: "+str(i)

#now calculate everything for the unmodified/unnormalized outputs
out_validation=[0]*N_epochs # output of net, when inputting x_mod of validation data and making the normalization of the output backwards, saved after each epoch
error_validation_data=[0.0]*N_epochs
diff_validation=[0.0]*N_epochs

#make the transformation on the outputs backwards
for i in range(0,N_epochs):
  out_validation[i]=np.ndarray(shape=(validation_set.get_N(),1))
  for j in range(0,len(out_mod_validation[i])):
    out_validation[i][j]=out_mod_validation[i][j]#do this, because otherwise we will produce only a reference

  readData.apply_inverse_function_to_outputs(out_mod_validation[i],out_validation[i],full_set.get_y_max())# second argument will be changed!

  diff_validation[i]=np.subtract(out_validation[i],validation_set.get_y())
  error_validation_data[i]=np.sum(np.absolute(np.subtract(out_validation[i],validation_set.get_y())))

#print at 10 examples how good the output matches the desired output
for i in range(0,10):
	print "desired output"
	print validation_set.get_y()[i][0]
	print "actual output after last training epoch"
	print out_validation[-1][i][0]
	print "-------"


print "total error on validation_data set after last training"
print error_validation_data[-1]

# # # # ## 
#printing#
#        #


plt.figure(1)
plt.title("Costfunction of (modified) Training-data")
plt.xlabel("epochs")
plt.ylabel("cost function")
x_range=[x+1 for x in range(0,N_epochs)]
plt.plot(x_range,cost_training_data)
plt.savefig("cost_on_training_data.png")

plt.figure(2)
plt.title("f data")
plt.xlabel("epochs")
plt.ylabel("total error on validation data")
x_range=[x+1 for x in range(0,N_epochs)]
plt.plot(x_range,error_validation_data)
plt.savefig("error_on_val_data.png")




error_on_validation_data_after_training = diff_validation[-1].reshape((1,validation_set.get_N()))			
hist=TH1D('hist',"Errors on val data after last training epoch",200,-10000,10000)
fill_hist(hist,error_on_validation_data_after_training[0])
canvas=TCanvas(); 
hist.GetXaxis().SetTitle("desired Chi^2- outputted Chi^2");
hist.Draw()
canvas.SaveAs('error_on_val_data_hist.png')
