import numpy as np


class kNN:#this is class for K nearest neighbour
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
    def euclid_distance(x1,x2): # here it is an static method for calculating the distance between two vectors 
                        #formula used is the euclidean distance i.e srt(summation(vec1-vec2)^2)
        return np.sqrt(np.sum((x1-x2)**2))
    def fit(self,x,y):
        self.xtrain=x
        self.ytrain=y
        
    def predict(self,x):
        distance=[(self.euclid_distance(x,xt),yt) for xt,yt in zip(self.xtrain,self.ytrain) ]
        distance.sort(key=lambda d:d[0])
        n_label=[l for (_,l) in distance[:self.K]] # takes the top K nearest matches 
        count=self.count(n_label)
        # print(n_label)
        # print(distance[:self.K])
        # print(count)
        most_common=max(count,key=count.get)
        return most_common