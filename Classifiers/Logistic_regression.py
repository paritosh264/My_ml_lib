import numpy as np
import matplotlib.pyplot as plt
class LogisticRegression:
    def __init__(self,lr=0.1,epoch=500):
        self.lr=lr
        self.epoch=epoch

    @staticmethod
    def sigmoid(eq): #this is sigmoid function and formula uses eulers constant for making any number represented between 0 and 1 
        return 1/(1+np.exp(-eq)) # formula is : 1/1+e^-x where e  is eulers constant =2.718... and x is some number
    
    def fit(self,x,y):
        x=np.array(x)
        y=np.array(y)
        if x.ndim==1:
              x=x.reshape(-1,1)
        x=np.insert(x,0,1,axis=1)
        sample_c,feature_c=x.shape
        self.w=np.ones(feature_c)
        for _ in range (self.epoch):

                regression=np.dot(self.w,x.T)  # even after being a classifier this model is named as regression because of this logic
                y_hat=self.sigmoid(regression)
                error=y-y_hat
                dl_dw=-1/sample_c*(np.dot(error.T,x))
                self.w-=self.lr*dl_dw
    '''now here as it is a binary classifier we need to use binary cross entropy loss for calculating the loss 
    binary cross entropy is summation y*log(error)-(1-y)*log(error)
    next we must find how much the wieghts and biases have contributed to this y_hat by backpropogation i.e by finding out derivative of 
    loss with respect to weight and biases . dy/dw and dy/db'''
    
    def predict(self,x):
         x=np.array(x)
         if x.ndim==1:
              x=x.reshape(-1,1)
         x = np.insert(x, 0, 1, axis=1) 
         regression=np.dot(x,self.w.T) 
         y_hat=self.sigmoid(regression)
         return y_hat

    


