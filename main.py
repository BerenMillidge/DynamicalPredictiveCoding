# trying to get predictive coding networks WORKING without the super strict issues on the constrains and amortisation etc.
#if this DID work it would be huge. Basically my PC capstone paper so I need to fiddle with it until it does, or at least understand WHY it doesn't

import numpy as np
import matplotlib.pyplot as plt
import torch 
import torchvision
import torchvision.transforms as transforms
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time
import matplotlib.pyplot as plt
import subprocess
import argparse
from datetime import datetime

num_batches= 10
num_train_batches=20
batch_size = 64

def get_dataset(batch_size,download):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./cifar_data', train=True,
                                            download=download, transform=transform)
    print("trainset: ", trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=1)
    print("trainloader: ", trainloader)
    trainset = list(iter(trainloader))

    testset = torchvision.datasets.CIFAR10(root='./cifar_data', train=False,
                                        download=download, transform=transform)
    print("testset: ", testset)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=1)
    print("testloader: ", testloader)
    testset = list(iter(testset))
    return trainset, testset


def boolcheck(x):
    return str(x).lower() in ["true", "1", "yes"]

def onehot(x):
    z = torch.zeros([len(x),10])
    for i in range(len(x)):
      z[i,x[i]] = 1
    return z.float().to(DEVICE)

def set_tensor(xs):
  return xs.float().to(DEVICE)

def tanh(xs):
    return torch.tanh(xs)

def linear(x):
    return x

def tanh_deriv(xs):
    return 1.0 - torch.tanh(xs) ** 2.0

def linear_deriv(x):
    return set_tensor(torch.ones((1,)))

def relu(xs):
  return torch.clamp(xs,min=0)

def relu_deriv(xs):
  rel = relu(xs)
  rel[rel>0] = 1
  return rel 

def softmax(xs):
  return torch.nn.softmax(xs)

def sigmoid(xs):
  return F.sigmoid(xs)

def sigmoid_deriv(xs):
  return F.sigmoid(xs) * (torch.ones_like(xs) - F.sigmoid(xs))
   
def edge_zero_pad(img,d):
  N,C, h,w = img.shape 
  x = torch.zeros((N,C,h+(d*2),w+(d*2))).to(DEVICE)
  x[:,:,d:h+d,d:w+d] = img
  return x


def accuracy(out, L):
  B,l = out.shape
  total = 0
  for i in range(B):
    if torch.argmax(out[i,:]) == torch.argmax(L[i,:]):
      total +=1
  return total/ B



class ConvLayer(object):
  def __init__(self,input_size,num_channels,num_filters,batch_size,kernel_size,learning_rate,f,df,padding=0,stride=1,device="cpu"):
    self.input_size = input_size
    self.num_channels = num_channels
    self.num_filters = num_filters
    self.batch_size = batch_size
    self.kernel_size = kernel_size
    self.padding = padding
    self.stride = stride
    self.output_size = math.floor((self.input_size + (2 * self.padding) - self.kernel_size)/self.stride) +1
    self.learning_rate = learning_rate
    self.f = f
    self.df = df
    self.device = device
    self.kernel= torch.empty(self.num_filters,self.num_channels,self.kernel_size,self.kernel_size).normal_(mean=0,std=0.05).to(self.device)
    self.unfold = nn.Unfold(kernel_size=(self.kernel_size,self.kernel_size),padding=self.padding,stride=self.stride).to(self.device)
    self.fold = nn.Fold(output_size=(self.input_size,self.input_size),kernel_size=(self.kernel_size,self.kernel_size),padding=self.padding,stride=self.stride).to(self.device)

  def forward(self,inp):
    self.X_col = self.unfold(inp.clone())
    self.flat_weights = self.kernel.reshape(self.num_filters,-1)
    out = self.flat_weights @ self.X_col
    self.activations = out.reshape(self.batch_size, self.num_filters, self.output_size, self.output_size)
    return self.f(self.activations)

  def update_weights(self,e,update_weights=False):
    fn_deriv = self.df(self.activations)
    e = e * fn_deriv
    self.dout = e.reshape(self.batch_size,self.num_filters,-1)
    dW = self.dout @ self.X_col.permute(0,2,1)
    dW = torch.sum(dW,dim=0)
    dW = dW.reshape((self.num_filters,self.num_channels,self.kernel_size,self.kernel_size))
    if update_weights:
      self.kernel += self.learning_rate * torch.clamp(dW * 2,-50,50)
    return dW

  def backward(self,e):
    fn_deriv = self.df(self.activations)
    e = e * fn_deriv
    self.dout = e.reshape(self.batch_size,self.num_filters,-1)
    dX_col = self.flat_weights.T @ self.dout
    dX = self.fold(dX_col)
    return torch.clamp(dX,-50,50)

  def get_true_weight_grad(self):
    return self.kernel.grad

  def set_weight_parameters(self):
    self.kernel = nn.Parameter(self.kernel)

  def save_layer(self,logdir,i):
      np.save(logdir +"/layer_"+str(i)+"_weights.npy",self.kernel.detach().cpu().numpy())

  def load_layer(self,logdir,i):
    kernel = np.load(logdir +"/layer_"+str(i)+"_weights.npy")
    self.kernel = set_tensor(torch.from_numpy(kernel))

class MaxPool(object):
  def __init__(self, kernel_size,device='cpu'):
    self.kernel_size = kernel_size
    self.device = device
    self.activations = torch.empty(1)

  def forward(self,x):
    out, self.idxs = F.max_pool2d(x, self.kernel_size,return_indices=True)
    return out
  
  def backward(self, y):
    return F.max_unpool2d(y,self.idxs, self.kernel_size)

  def update_weights(self,e,update_weights=False):
    return 0

  def get_true_weight_grad(self):
    return None

  def set_weight_parameters(self):
    pass

  def save_layer(self,logdir,i):
    pass

  def load_layer(self,logdir,i):
    pass

class AvgPool(object):
  def __init__(self, kernel_size,device='cpu'):
    self.kernel_size = kernel_size
    self.device = device
    self.activations = torch.empty(1)
  
  def forward(self, x):
    self.B_in,self.C_in,self.H_in,self.W_in = x.shape
    return F.avg_pool2d(x,self.kernel_size)

  def backward(self, y):
    N,C,H,W = y.shape
    print("in backward: ", y.shape)
    return F.interpolate(y,scale_factor=(1,1,self.kernel_size,self.kernel_size))

  def update_weights(self,x):
    return 0

  def save_layer(self,logdir,i):
    pass

  def load_layer(self,logdir,i):
    pass


class ProjectionLayer(object):
  def __init__(self,input_size, output_size,f,df,learning_rate,device='cpu'):
    self.input_size = input_size
    self.B, self.C, self.H, self.W = self.input_size
    self.output_size =output_size
    self.learning_rate = learning_rate
    self.f = f
    self.df = df
    self.device = device
    self.Hid = self.C * self.H * self.W
    self.weights = torch.empty((self.Hid, self.output_size)).normal_(mean=0.0, std=0.05).to(self.device)

  def forward(self, x):
    self.inp = x.detach().clone()
    out = x.reshape((len(x), -1))
    self.activations = torch.matmul(out,self.weights)
    return self.f(self.activations)

  def backward(self, e):
    fn_deriv = self.df(self.activations)
    out = torch.matmul(e * fn_deriv, self.weights.T)
    out = out.reshape((len(e), self.C, self.H, self.W))
    return torch.clamp(out,-50,50)

  def update_weights(self, e,update_weights=False):
    out = self.inp.reshape((len(self.inp), -1))
    fn_deriv = self.df(self.activations)
    dw = torch.matmul(out.T, e * fn_deriv)
    if update_weights:
      self.weights += self.learning_rate * torch.clamp((dw * 2),-50,50)
    return dw

  def get_true_weight_grad(self):
    return self.weights.grad

  def set_weight_parameters(self):
    self.weights = nn.Parameter(self.weights)

  def save_layer(self,logdir,i):
    np.save(logdir +"/layer_"+str(i)+"_weights.npy",self.weights.detach().cpu().numpy())

  def load_layer(self,logdir,i):
    weights = np.load(logdir +"/layer_"+str(i)+"_weights.npy")
    self.weights = set_tensor(torch.from_numpy(weights))

  def save_layer(self,logdir,i):
    np.save(logdir +"/layer_"+str(i)+"_weights.npy",self.weights.detach().cpu().numpy())

  def load_layer(self,logdir,i):
    weights = np.load(logdir +"/layer_"+str(i)+"_weights.npy")
    self.weights = set_tensor(torch.from_numpy(weights))

class FCLayer(object):
  def __init__(self, input_size,output_size,batch_size, learning_rate,f,df,device="cpu"):
    self.input_size = input_size
    self.output_size = output_size
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.f = f 
    self.df = df
    self.device = device
    self.weights = torch.empty([self.input_size,self.output_size]).normal_(mean=0.0,std=0.05).to(self.device)

  def forward(self,x):
    self.inp = x.clone()
    self.activations = torch.matmul(self.inp, self.weights)
    return self.f(self.activations)

  def backward(self,e):
    self.fn_deriv = self.df(self.activations)
    out = torch.matmul(e * self.fn_deriv, self.weights.T)
    return torch.clamp(out,-50,50)

  def update_weights(self,e,update_weights=False):
    self.fn_deriv = self.df(self.activations)
    dw = torch.matmul(self.inp.T, e * self.fn_deriv)
    if update_weights:
      self.weights += self.learning_rate * torch.clamp(dw*2,-50,50)
    return dw

  def get_true_weight_grad(self):
    return self.weights.grad

  def set_weight_parameters(self):
    self.weights = nn.Parameter(self.weights)

class PCNet(object):
  def __init__(self, layers, n_inference_steps_train, inference_learning_rate, weight_learning_rate,with_amortisation=False,continual_weight_update=False,update_dilation_factor=None,numerical_check=False,device='cpu'):
    self.layers= layers
    self.n_inference_steps_train = n_inference_steps_train
    self.inference_learning_rate = inference_learning_rate
    self.weight_learning_rate = weight_learning_rate
    self.device = device
    self.L = len(self.layers)
    self.outs = [[] for i in  range(self.L+1)]
    self.prediction_errors = [[] for i in range(self.L+1)]
    self.predictions = [[] for i in range(self.L+1)]
    self.mus = [[] for i in range(self.L+1)]
    self.with_amortisation = with_amortisation
    self.continual_weight_update = continual_weight_update
    self.numerical_check = numerical_check
    self.update_dilation_factor = update_dilation_factor if update_dilation_factor is not None else self.n_inference_steps_train
    if self.continual_weight_update:
      for l in self.layers:
        if hasattr(l, "learning_rate"):
          l.learning_rate = l.learning_rate / self.update_dilation_factor

    if self.numerical_check:
      for l in self.layers:
        l.set_weight_parameters()

  def update_weights(self,print_weight_grads=False,get_errors=False):
    weight_diffs = []
    for (i,l) in enumerate(self.layers):
      if i !=1:
        dW = l.update_weights(self.prediction_errors[i+1],update_weights=True)
        if print_weight_grads:
          print("weight diffs: ",(dW*2) + true_weight_grad)
          diff = torch.sum((dW -true_dW)**2)
          weight_diffs.append(diff)
    return weight_diffs

  def forward(self,x):
    for i,l in enumerate(self.layers):
      x = l.forward(x)
    return x

  def no_grad_forward(self,x):
    with torch.no_grad():
      for i,l in enumerate(self.layers):
        x = l.forward(x)
      return x

  def infer(self, inp,label,n_inference_steps=None,fixed_predictions=False,test=False):
    self.n_inference_steps_train = n_inference_steps if n_inference_steps is not None else self.n_inference_steps_train
    with torch.no_grad():
      self.mus[0] = inp.clone()
      self.outs[0] = inp.clone()
      #predictions
      for i,l in enumerate(self.layers):
        #initialize mus with forward predictions)
        if self.with_amortisation:
          self.outs[i+1] = l.forward(self.outs[i])
          self.mus[i+1] = self.outs[i+1].clone()
        else:
          #print("mus: ", self.mus[i][0,:])
          self.outs[i+1] = l.forward(self.mus[i].clone())
          self.mus[i+1] = set_tensor(torch.empty_like(self.outs[i+1]).normal_(mean=0,std=0.05)) 
      if test is False:
        self.mus[-1] = label.clone() #setup final label
        #print("mus at beginning: ", self.mus[-1])
        self.prediction_errors[-1] = self.mus[-1] - self.outs[-1] #setup final prediction errors
      else:
        self.prediction_errors[-1] = self.mus[-1] - self.outs[-1]
      #self.predictions[-1] = self.prediction_errors[-1].clone()
      for n in range(self.n_inference_steps_train):
        self.mus[0] = inp.clone()
        self.outs[0] = inp.clone()
        if not fixed_predictions:
          for i,l in enumerate(self.layers):
            self.outs[i+1] = l.forward(self.mus[i].clone())
          self.prediction_errors[-1] = self.mus[-1] - self.outs[-1] #setup final prediction errors
          #self.predictions[-1] = self.prediction_errors[-1].clone()
        for j in reversed(range(len(self.layers))):
          self.prediction_errors[j] = self.mus[j] - self.outs[j]
          self.predictions[j] = self.layers[j].backward(self.prediction_errors[j+1])
          dx_l = self.prediction_errors[j] - self.predictions[j]

          self.mus[j] -= self.inference_learning_rate * (2*dx_l)
        
        if self.continual_weight_update:
          weight_diffs = self.update_weights()

      if test:
        return self.mus[-1]
      weight_diffs = self.update_weights()
      L = torch.sum(self.prediction_errors[-1]**2).item()
      print("predictions: ", self.outs[-1][0,:])
      print("mus: ", self.mus[-1][0,:])
      #get accuracy
      #train_acc = accuracy(self.outs[-1],label)
      acc = accuracy(self.no_grad_forward(inp),label)
      #test_acc_infer = accuracy(self.infer(inp, label,n_inference_steps=n_inference_steps, fixed_predictions=fixed_predictions, test=True),label)
      #print("Train Acc: ", train_acc)
      #print("Test Acc : ", acc)
      #print("Test Infer Acc : ", test_acc_infer)
      return L,self.mus[-1],acc

  def test_accuracy(self,testset):
    testaccs = []
    for i,(inp, label) in enumerate(testset):
        L, ypred,acc = self.infer(inp.to(DEVICE),onehot(label).to(DEVICE),fixed_predictions=fixed_predictions)
        testaccs.append(acc)
    return np.mean(np.array(testaccs))

  def save_model(self, savedir, logdir, losses,accs,test_accs):
    for i,l in enumerate(self.layers):
        l.save_layer(logdir,i)
    np.save(logdir +"/losses.npy",np.array(losses))
    np.save(logdir+"/accs.npy",np.array(accs))
    np.save(logdir+"/test_accs.npy",np.array(test_accs))
    subprocess.call(['rsync','--archive','--update','--compress','--progress',str(logdir) +"/",str(savedir)])
    print("Rsynced files from: " + str(logdir) + "/ " + " to" + str(savedir))
    now = datetime.now()
    current_time = str(now.strftime("%H:%M:%S"))
    subprocess.call(['echo','saved at time: ' + str(current_time)])

  def load_model(self,old_savedir):
    for (i,l) in enumerate(self.layers):
        l.load_layer(old_savedir,i)
        
  def train(self,trainset, testset,logdir, savedir,n_epochs,n_inference_steps,fixed_predictions=False):
    with torch.no_grad():
        losses = []
        accs = []
        test_accs = []
        for epoch in range(n_epochs):
            print("Epoch: ", epoch)
            losslist = []
            acclist = []
            for i,(inp, label) in enumerate(trainset):
                L, ypred,acc = self.infer(inp.to(DEVICE),onehot(label).to(DEVICE),fixed_predictions=fixed_predictions)
                losslist.append(L)
                acclist.append(acc)

            losses.append(np.mean(np.array(losslist)))
            accs.append(np.mean(np.array(acclist)))
            test_accs.append(test_accuracy(testset))
            self.save_model(logdir,savedir, losses,accs,test_accs)

if __name__ == '__main__':
    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    print("Initialized")
    #parsing arguments
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--savedir",type=str,default="savedir")
    parser.add_argument("--batch_size",type=int, default=64)
    parser.add_argument("--learning_rate",type=float,default=0.0005)
    parser.add_argument("--N_epochs",type=int, default=1000)
    parser.add_argument("--save_every",type=int, default=1)
    parser.add_argument("--old_savedir",type=str,default="None")
    parser.add_argument("--n_inference_steps",type=int,default=100)
    parser.add_argument("--inference_learning_rate",type=float,default=0.1)
    parser.add_argument("--dataset",type=str,default="mnist")
    parser.add_argument("--with_amortisation",type=boolcheck, default=True)
    parser.add_argument("--continual_weight_update",type=boolcheck, default=False)
    parser.add_argument("--fixed_predictions",type=boolcheck,default=False)
    parser.add_argument("--download_data",type=boolcheck,default=False)
    args = parser.parse_args()
    print("Args parsed")
    #create folders
    if args.savedir != "":
        subprocess.call(["mkdir","-p",str(args.savedir)])
    if args.logdir != "":
        subprocess.call(["mkdir","-p",str(args.logdir)])
    print("folders created")
    trainset,testset = get_dataset(args.batch_size,args.download_data)
    l1 = ConvLayer(32,3,6,64,5,args.learning_rate,relu,relu_deriv,device=DEVICE)
    l2 = MaxPool(2,device=DEVICE)
    l3 = ConvLayer(14,6,16,64,5,args.learning_rate,relu,relu_deriv,device=DEVICE)
    l4 = ProjectionLayer((64,16,10,10),120,relu,relu_deriv,args.learning_rate,device=DEVICE)
    l5 = FCLayer(120,84,64,args.learning_rate,relu,relu_deriv,device=DEVICE)
    l6 = FCLayer(84,10,64,args.learning_rate,linear,linear_deriv,device=DEVICE)
    layers =[l1,l2,l3,l4,l5,l6]
    net = PCNet(layers,args.n_inference_steps,args.inference_learning_rate,args.learning_rate,with_amortisation=args.with_amortisation,continual_weight_update=args.continual_weight_update,update_dilation_factor=200,device=DEVICE)
    net.train(trainset[0:-2],testset[0:-2],args.logdir, args.savedir,args.N_epochs, args.n_inference_steps,fixed_predictions=args.fixed_predictions)