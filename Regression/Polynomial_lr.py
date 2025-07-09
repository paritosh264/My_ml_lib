import numpy as np
 

class PolynomialLr:
    def __init__(self,degree,lrate=0.01,epochs=500,grad_desc=True):
        self.deg=degree
        self.lr=lrate
        self.grd=grad_desc
        self.epoch=epochs

    def degree(self,x):
        newf=[np.ones((x.shape[0],1))]
        for d in range(1,self.deg+1):
            newf.append(x**d)

            
        return np.hstack(newf)

    def fit(self,x,y):
        x=np.array(x)
        y=np.array(y)
        if x.ndim<2:
            x=x.reshape(-1,1)
        if self.grd:
        
                self.mean=np.mean(x,axis=0)
                self.std=np.std(x,axis=0)+1e-8
                x=(x-self.mean)/self.std
                x=self.degree(x)
                samples,features=x.shape
                self.weights=np.ones(features)
                for _ in range(self.epoch):
                    y_hat= np.dot(x,self.weights)
                    dl_dw=-2/samples*np.dot(x.T,(y-y_hat))
                    self.weights-=self.lr*dl_dw
        else:
                 x=self.degree(x)
                 self.weights=np.linalg.inv(x.T@x)@x.T@y

        


    


    def predict(self,x):
        x=np.array(x)
        if x.ndim<2:
            x=x.reshape(-1,1)
        x=(x-self.mean)/self.std
        x=self.degree(x)
        y_hat= np.dot(x,self.weights)
        return y_hat

                




