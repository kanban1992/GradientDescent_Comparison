import numpy as np
import root_numpy
from ROOT import TFile, TH2D, TCanvas
import itertools


def apply_function_to_inputs(x,x_mod,x_max):# python uebergibt alles als reference
   #normalize the inputs
   for i in range(0,len(x)):
      for j in range(0,len(x[i])):
        #print "x["+str(i)+"]["+str(j)+"]="+str(x[i][j])
        x_mod[i][j]=x[i][j]/x_max[j]
        #print "x_mod["+str(i)+"]["+str(j)+"]="+str(x_mod[i][j])

def apply_inverse_function_to_inputs(x,x_mod,x_max):# python uebergibt alles als reference
   #re normalize the inputs
   for i in range(0,len(x)):
      for j in range(0,len(x[i])):
        x_mod[i][j]=x[i][j]*x_max[j]

def apply_function_to_outputs(y,y_mod,y_max):# python uebergibt alles als reference
   #normalize the outputs
   for i in range(0,len(y)):
      for j in range(0,len(y[i])):
        y_mod[i][j]=y[i][j]/y_max[j]

def apply_inverse_function_to_outputs(y,y_mod,y_max):# python uebergibt alles als reference
   #re-normalize the outputs
   for i in range(0,len(y)):
      for j in range(0,len(y[i])):
        y_mod[i][j]=y[i][j]*y_max[j]

class Dataset(object):
  def __init__(self,path,hist_name,kind_of_set,function_inputs,function_outputs,full_set,subset):
    
    self._kind_of_set=kind_of_set     
    
    """example
    self._x np.ndarray(shape=(N_points,2))
    [[   10.    95.]
     [   10.   100.]
     [   10.   105.]
     ..., 
     [ 2490.  1185.]
     [ 2490.  1190.]
     [ 2490.  1195.]]


    self._y np.ndarray(shape=(N_points,1))
    [[  0.00000000e+00]
     [  0.00000000e+00]
     [  0.00000000e+00]
     ..., 
     [  6.34848448e-06]
     [  6.34845946e-06]
     [  6.34848448e-06]]
    """

    rfile = TFile(path)
    histogram = rfile.Get(hist_name)

    #now prepare data for training:
    if kind_of_set=="full_set":
        N_points=histogram.GetXaxis().GetNbins() * histogram.GetYaxis().GetNbins() #number of points in full_set
        self._N=N_points
        self._y=np.ndarray(shape=(N_points,1))
        self._x=np.ndarray(shape=(N_points,2))
        self._y_mod=np.ndarray(shape=(N_points,1)) #function applied to outputs, for example normalized, or a function is applied
        self._x_mod=np.ndarray(shape=(N_points,2)) #function applied to inputs
        self._y_max=np.ndarray(shape=(1))
        self._y_max[0]=0.0
        self._x_max=np.ndarray(shape=(2))
        self._x_max=np.ndarray(shape=(2))
	self._x_max[0]=0.0
	self._x_max[1]=0.0
        i=0
        for x_bin in range(0, histogram.GetXaxis().GetNbins()):
            for y_bin in range(0, histogram.GetYaxis().GetNbins()):
                self._x[i][0]=histogram.GetXaxis().GetBinCenter(x_bin)
                self._x[i][1]=histogram.GetYaxis().GetBinCenter(y_bin)
                self._y[i][0]=histogram.GetBinContent(x_bin,y_bin)
                for j in range(0,len(self._x[i])):# only in the full_set case the maximum values are calculated
                  if self._x[i][j]>self._x_max[j]:
                    self._x_max[j]=self._x[i][j]
                for j in range(0,len(self._y[i])):
                  if self._y[i][j]>self._y_max[j]:
                    self._y_max[j]=self._y[i][j]
                i=i+1
	#apply function to inputs and outputs, the function can also be the identity
	apply_function_to_inputs(self._x,self._x_mod,self._x_max)
	apply_function_to_outputs(self._y,self._y_mod,self._y_max)
      
               
    elif kind_of_set=="training_set" or kind_of_set=="validation_set" or kind_of_set=="test_set":
        self._N = len(subset)#Number of elements of the data set
        self._y=np.ndarray(shape=(self._N,1))
        self._x=np.ndarray(shape=(self._N,2))
        self._y_mod=np.ndarray(shape=(self._N,1))
        self._x_mod=np.ndarray(shape=(self._N,2))
        self._y_max=full_set.get_y_max()
        self._x_max=full_set.get_x_max()
        for i in range(0,self._N):
            self._x[i][0]=full_set.get_x()[subset[i]][0]
            self._x[i][1]=full_set.get_x()[subset[i]][1]
            self._y[i][0]=full_set.get_y()[subset[i]][0]
            self._x_mod[i][0]=full_set.get_x_mod()[subset[i]][0]
            self._x_mod[i][1]=full_set.get_x_mod()[subset[i]][1]
            self._y_mod[i][0]=full_set.get_y_mod()[subset[i]][0]	    



    

    if len(self._x)==0:# If the set has 0 entries the list  is empty
      self._N_input=-1
    else:
      self._N_input = len(self._x[0])

    if len(self._y)==0:# If the set has 0 entries the list  is empty
      self._N_output=-1
    else:
      self._N_output = len(self._y[0])

    self._index_in_epoch = 0 #if one has trained 2 mini batches in the epoch already then this is 2*batch_size
    self._epochs_completed = 0


  def get_N_input_nodes(self):
    return self._N_input

  def get_N_output_nodes(self):
    return self._N_output

  def get_N(self):
    return self._N

  def get_x(self):
    return self._x

  def get_y(self):
    return self._y

  def get_x_max(self):
    return self._x_max

  def get_y_max(self):
    return self._y_max

  def get_x_mod(self):
    return self._x_mod
  
  def get_y_mod(self):
    return self._y_mod

  def next_batch(self, batch_size, fake_x=False):
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch >= self._N:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._N)
      np.random.shuffle(perm)
      self._x = self._x[perm]#shuffle both, actually one would only need to shuffle x_mod and y_mod, but for consistency we shuffle both!
      self._y = self._y[perm]
      self._x_mod = self._x_mod[perm]
      self._y_mod = self._y_mod[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._N #if batch size<= self._N then an exception is thrown!
    end = self._index_in_epoch
    return self._x_mod[start:end], self._y_mod[start:end], self._epochs_completed


def read_data_set(path,hist_name,kind_of_set,function_inputs,function_outputs,full_set=None,subset=None):
  return Dataset(path,hist_name,kind_of_set,function_inputs,function_outputs,full_set,subset)

