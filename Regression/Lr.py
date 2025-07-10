import numpy as np
import matplotlib.pyplot as plt
class Linear_reg:
    def __init__(self,learning_rate=0.01,epoch=500):
        self.lr=learning_rate
        self.epoch=epoch
        self.loss=[] #this is just to store the losses fr plotiing , it could be ignored...
        

    def fit(self,x,y,gd=True):
        x=np.array(x)
        y=np.array(y)
        n=len(x)
        self.weight=np.zeros(x.shape[1]) # creating the weights vector for number of features 
        self.bias=0.0 #bias could be scalar
        if gd:
            for _ in range (self.epoch):
                y_hat=np.dot(x,self.weight)+self.bias # y_hat is the prediction for every sample
                dw=-2/n*np.dot(x.T,(y-y_hat)) # here mean squared error is used for calculating the error , although we only need the derivative
                db=-2/n*np.sum(y-y_hat)        # of the mse with respect to weights and biases , represented by dw and db here for gradient descent
                self.weight-=self.lr*dw
                self.bias-=self.lr*db
                me=np.mean((y-y_hat)**2) # this is the mse (error) just for plotting its just optional
                self.loss.append(me)
        else:
                 self.weights=np.linalg.inv(x.T@x)@x.T@y

            
    def predict(self,x):
        return np.dot(x,self.weight)+self.bias # once weights and biases are found we can predict by using them 
    

    def plot_loss(self):
        plt.plot(range(1, self.epoch + 1), self.loss, color='blue')
        plt.title('Training Loss (MSE) Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.grid(True)
        plt.show()
