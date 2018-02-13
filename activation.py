import numpy as np

class Activation:
    def result(self,x):
        pass

    def prime(self,x):
        pass


class Sigmoid(Activation):
    def result(self,x):
        return 1.0/(1.0+np.exp(-x))

    def prime(self,x):
        return self.result(x)*(1 - self.result(x))


class Relu(Activation):
    def result(self,x):
        return np.maximum(x,0)

    def prime(self,x):
        return np.where(x>0,1,0)

class Tanh(Activation):
    def result(self,x):
        return 2.0/(1+np.exp(-2*x)) - 1

    def prime(self,x):
        return 1 - np.power(self.result(x),2)
