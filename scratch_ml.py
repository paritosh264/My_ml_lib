import numpy as np
# import sklearn
import matplotlib.pyplot as plt

class Linear_reg:
    def __init__(self,learning_rate=0.01,epoch=500):
        self.lr=learning_rate
        self.epoch=epoch
        self.loss=[]

    def fit(self,x,y):
        x=np.array(x)
        y=np.array(y)
        n=len(x)
        self.weight=np.zeros(x.shape[1])
        self.bias=0.0
        
        for _ in range (self.epoch):
            y_hat=np.dot(x,self.weight)+self.bias
            dw=-2/n*np.dot(x.T,(y-y_hat))
            db=-2/n*np.sum(y-y_hat)
            self.weight-=self.lr*dw
            self.bias-=self.lr*db
            me=np.mean((y-y_hat)**2)
            self.loss.append(me)

            
    def predict(self,x):
        return np.dot(x,self.weight)+self.bias
    

    def plot_loss(self):
        plt.plot(range(1, self.epoch + 1), self.loss, color='blue')
        plt.title('Training Loss (MSE) Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.grid(True)
        plt.show()


class kNN:
    def __init__(self,k=3):
        self.K=k
        self.nn=[]
    @staticmethod
    def count(xlist):
        count={}
        
        for item in xlist:
            if item in count:
                count[item]+=1
            else:
                count[item]=1
        
        return count
    
    @staticmethod
    def euclid_distance(x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))
    def fit(self,x,y):
        self.xtrain=x
        self.ytrain=y
        
    def predict(self,x):
        distance=[(self.euclid_distance(x,xt),yt) for xt,yt in zip(self.xtrain,self.ytrain) ]
        distance.sort(key=lambda d:d[0])
        n_label=[l for (_,l) in distance[:self.K]]
        count=self.count(n_label)
        # print(n_label)
        # print(distance[:self.K])
        # print(count)
        most_common=max(count,key=count.get)
        return most_common
