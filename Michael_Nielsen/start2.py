import data_loader
import network2
import matplotlib.pyplot as plt
import ROOT 
from ROOT import TH1D, TCanvas, TH2D, TFile, TLegend
from root_numpy import fill_hist
import numpy as np


""" Verbesserungen gegenueber network.py sind
Cross entropy cost function anstatt der quadratic cost function ( vermeidung lerning slow down in last layer)
L2 regularization
other weight initialization to avoid saturation of first layer neurons

ideas not implemented:
dropout
Hessian technique fuer grdient calculation
other activation function : rectified linear neurons, tanh()...

learning-rate- schedule, learning rate halves if validation_accuracy satisfies the no-improvement_in_20 rule and terminater if the learning rate has 1/128 of it's original value.
momentum based gradient descent
"""


file_directory="."
file_name="TH2D_A00_TB10.root"
hist_name="LHCChi2_CMSSM_nObs1061_A00_TB10"

full_data,training_data,test_data=data_loader.load_data(10000,file_directory,file_name,hist_name)

#structure of full_data [(np.array([[m0],[m12]]),np.array([[chi^2]])),(np.array([[m0],[m12]]),np.array([[chi^2]])),...]
#structure of training_data : as full_data
#structure of test_data     [(np.array([[m0],[m12]]),chi^2),(np.array([[m0],[m12]]),chi^2),...]

num_inputs=2
num_outputs=1

max_inputs=[0]*num_inputs
max_outputs=[0]*num_outputs


#first normalize full_data set! This will automatically normalize the training data set(see data_loader). This also normalizes the input of test_data, but not the output 
#of test data, because this is saved not as numpy array, but as plain number!

for i in range(0,len(full_data)):
	if max_inputs[0]< full_data[i][0][0][0]:
		max_inputs[0]=full_data[i][0][0][0]
	if max_inputs[1]<full_data[i][0][1][0]:
		max_inputs[1]=full_data[i][0][1][0]
	if max_outputs[0]<full_data[i][1][0][0]:
		max_outputs[0]=full_data[i][1][0][0]

for i in range(0,len(full_data)):
	full_data[i][0][0][0]=full_data[i][0][0][0]/max_inputs[0]
	full_data[i][0][1][0]=full_data[i][0][1][0]/max_inputs[1]
	full_data[i][1][0][0]=full_data[i][1][0][0]/max_outputs[0]

# normalize test_data output
for i in range(0,len(test_data)):
	lst=list(test_data[i])
	lst[1]=test_data[i][1]/max_outputs[0]
	t=tuple(lst)
	test_data[i]=t
# I have checked that the normalization works well!




#-----------------------------------------------------
#Hyper parameters
epochs=20
mini_batch_size=10
eta=3.0
lm_da=0
cost_function="quadratic"# choices are "quadratic" or "crossEntropy"
weight_initialization="small"# choices are "large" , "small"
#-----------------------------------------------------
#initialization
net=None
if cost_function=="quadratic" and weight_initialization=="small":
	net = network2.Network([num_inputs, 30,30,30, num_outputs], cost=network2.QuadraticCost)
elif cost_function=="quadratic" and weight_initialization=="large":
	net = network2.Network([num_inputs, 30,30,30, num_outputs], cost=network2.QuadraticCost)
	net.large_weight_initializer()
elif cost_function=="crossEntropy" and weight_initialization=="small":
	net = network2.Network([num_inputs, 30,30,30, num_outputs], cost=network2.CrossEntropyCost)
elif cost_function=="crossEntropy" and weight_initialization=="large":
	net = network2.Network([num_inputs, 30,30,30, num_outputs], cost=network2.CrossEntropyCost)
	net.large_weight_initializer()

evaluation_cost, evaluation_error, training_cost, training_error, evaluation_errors_last_epoch=net.SGD(training_data, epochs, mini_batch_size, eta, lmbda =lm_da,evaluation_data=test_data,monitor_evaluation_cost=True, monitor_evaluation_error=True,monitor_training_cost=True,monitor_training_error=True)


#-----------------------------------------------------
#Plotting


#plot original data
path = file_directory + "/" + file_name
rfile = TFile(path)
histogram = rfile.Get(hist_name)

canvas = TCanvas("canvas", "canvas", 1000, 600)
histogram.GetXaxis().SetTitle("M_0")
histogram.GetYaxis().SetTitle("M_1/2")
histogram.Draw("COLZ")
ROOT.gPad.Update()
statbox_temp_eval_reg = histogram.FindObject("stats")#set position of stat box
statbox_temp_eval_reg.SetX1NDC(0.0)
statbox_temp_eval_reg.SetX2NDC(0.1)
statbox_temp_eval_reg.SetY1NDC(0.9)
statbox_temp_eval_reg.SetY2NDC(1.0)
canvas.Print("./data.png")

x_min=5
y_min=90
# |-----|-----|-----|-----|-----|-----|-----|  binwidth=10
# 0     5    10    15    20    25    30    35    
#     x_min  x[0]        x[1]        x[2]
#create a histogram that shows the result after training.
histogram_training_results=histogram.Clone()
histogram_training_results_rel_error=histogram.Clone()
histogram_training_results_abs_error=histogram.Clone()

bin_width_x=histogram_training_results.GetXaxis().GetBinWidth(1)
bin_width_y=histogram_training_results.GetYaxis().GetBinWidth(1)
print bin_width_x
print bin_width_y

for (x,y) in full_data:
	output=net.feedforward(x)[0][0]
	rel_error=output
	abs_error=abs(output-y[0][0])
	if y[0][0]!=0.0:
		rel_error=output/y[0][0]
		#print "yes"
	x_bin= int(round((x[0][0]*max_inputs[0]-x_min-0.5*bin_width_x)/bin_width_x))#round() to nearest int number
	y_bin= int(round((x[1][0]*max_inputs[1]-y_min-0.5*bin_width_y)/bin_width_y))
	histogram_training_results_rel_error.SetBinContent(x_bin,y_bin,rel_error)
	histogram_training_results_rel_error.SetBinContent(x_bin,y_bin,abs_error)
	histogram_training_results.SetBinContent(x_bin, y_bin, output*max_outputs[0])

j=0
#print the input, the correspondinf output and corresponding desired output
for (x,y) in test_data:
	if j>20:
		break
	print "x"
	print x
	print "y (output of net)"
	print net.feedforward(x)[0][0]
	print "y (desired)"
	print y 
	print "-----------------------"
	j=j+1

canvas1 = TCanvas("canvas1", "canvas1", 1000, 600)
histogram_training_results.Draw("COLZ")
canvas1.Print("./data_after_training.png")



#plot error on evaluation_data without normalization


evaluation_error_no_norm=[x*max_outputs[0] for x in evaluation_error]
evaluation_errors_last_epoch_no_norm=[x*max_outputs[0] for x in evaluation_errors_last_epoch]

titlename="eta="+str(eta)+", mini_batch_size="+str(mini_batch_size)+ ", lambda="+str(lm_da)+",\n Cost="+cost_function+", weight_initialization="+weight_initialization
titlename_no_norm=titlename+"_without_normalization"

plt.figure(2)

plt.title(titlename_no_norm)
plt.xlabel("epochs")
plt.ylabel("total error validation data")
x_range=[x+1 for x in range(0,epochs)]
plt.plot(x_range,evaluation_error_no_norm)
plt.savefig("error_on_val_data_"+titlename_no_norm+".png")

print "total error on validation_data set after last training"
print evaluation_error_no_norm[-1]
					
hist_no_norm=TH1D('hist_no_norm',titlename_no_norm,50,-100,100)
fill_hist(hist_no_norm,evaluation_errors_last_epoch_no_norm)
canvas=TCanvas(); 
hist_no_norm.GetXaxis().SetTitle("desired Chi^2- outputted Chi^2");
hist_no_norm.Draw()
canvas.SaveAs('error_on_val_data_'+titlename_no_norm+'_hist.png')
#-----------------------------------------------------
