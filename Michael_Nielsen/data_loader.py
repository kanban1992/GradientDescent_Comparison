from ROOT import TFile, TH2D, TCanvas
import numpy as np
from random import randint


def load_data(num_test_points,file_directory,file_name,hist_name): 
	#access this with TBrowser()
	path = file_directory + "/" + file_name
	rfile = TFile(path)
	histogram = rfile.Get(hist_name)
		
	for x_bin in range(0, histogram.GetXaxis().GetNbins()):
		for y_bin in range(0, histogram.GetYaxis().GetNbins()):
			x = histogram.GetXaxis().GetBinCenter(x_bin)
			y = histogram.GetYaxis().GetBinCenter(y_bin)
			val=histogram.GetBinContent(x_bin,y_bin)
			#print str(x)+" "+str(y)+" "+str(val)


	#now prepare data for training:
	num_total_points=histogram.GetXaxis().GetNbins() * histogram.GetYaxis().GetNbins() #number of total points
	
	#input_points: for cMSSM2: [np.array([[10],[20]]),np.array([[10],[100]]),...]
	input_points=[np.array([[histogram.GetXaxis().GetBinCenter(x_bin)],[histogram.GetYaxis().GetBinCenter(y_bin)]]) for x_bin in range(0,histogram.GetXaxis().GetNbins()) for y_bin in range(0,histogram.GetYaxis().GetNbins())]
	
	#output_points: for cMSSM2: [np.array([[1.2]]),np.array([[3.2]]),...]
	output_points=[np.array([[histogram.GetBinContent(x_bin,y_bin)]]) for x_bin in range(0,histogram.GetXaxis().GetNbins()) for y_bin in range(0,histogram.GetYaxis().GetNbins())]

	#prepare a full_data file (which contains training and test data)
	full_data=zip(input_points,output_points)

	#print "length of total points_result array: "+ str(len(output_points))

	test_data=range(0,num_test_points)
	
	for i in range(0,num_test_points):
		rand_num=randint(0,num_total_points-1)
		#test_data structure: [(np.array([[10],[20]]),1.2),(np.array([[10],[100]]),3.2),next tuple]
		test_data[i]=(input_points[rand_num],output_points[rand_num][0][0])
		del input_points[rand_num]
		del output_points[rand_num]
		num_total_points=num_total_points-1
	#scan_data structure: [(np.array([[10],[20]]),np.array([[1.2]])),(np.array([[10],[100]]),np.array([[3.2]])),next tuple...]
	training_data=zip(input_points,output_points)
	return full_data,training_data, test_data 
