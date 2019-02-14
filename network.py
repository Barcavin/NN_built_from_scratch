import numpy as np
import random
import pickle

class NeuralNetwork:
    def __init__(self,layers,Acts):
        if (len(layers)-1!=len(Acts)):
            print("Wrong input!! #Layers = #Activations + 1")
            return
        self.layers = layers
        self.weights = [np.random.normal(0,0.1,(x,y)) for x,y in zip(layers[1:],layers[:-1])]
        self.biases = [np.random.normal(0,0.1,(x,1)) for x in layers[1:]]
        self.Activation = Acts

    def feedforward(self,x):
        for w,b,a in zip(self.weights,self.biases,self.Activation):
            x = a.result(np.dot(w,x)+b)
        result = np.exp(x)/np.sum(np.exp(x))
        return x

    def backprop(self,x,y):
        """
            Helper Function. Input a single sample and calculate the gradient.
        """
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        a_s = [x] # a_1(x),a_2,....,a_l
        z_s = [] # z_2,z_3,...,z_l
        delta = [] #delta_1,delta_2,...delta_l
        #Feedforward
        for w,b,a in zip(self.weights,self.biases,self.Activation):
            #print(w.shape)
            #print(a_s[-1].shape)
            #print(b.shape)
            z = np.dot(w,a_s[-1])+b
            z_s.append(z)
            a_s.append(a.result(z))

        y_hat = np.exp(a_s[-1])/np.sum(np.exp(a_s[-1]))

        #backprop
        #The calcualtion of the error term in the output layer is different from the one in the hidden layer.
        temp_delta = -(y-y_hat)**self.Activation[-1].prime(z_s[-1])
        delta = [temp_delta] + delta

        nabla_w[-1] = np.dot(delta[-1],a_s[-2].transpose())
        nabla_b[-1] = delta[-1]

        for l in range(2,len(self.layers)):
            temp_delta = np.dot(self.weights[-l+1].transpose(), delta[-l+1]) * self.Activation[-l].prime(z_s[-l])
            delta = [temp_delta] + delta
            nabla_w[-l] = np.dot(delta[-l],a_s[-l-1].transpose())
            nabla_b[-l] = delta[-l]

        return nabla_w , nabla_b

    def update(self,batch,learning_rate,regular):
        """
            Helper Function: input a batch of samples. Update the weight matrices
        """
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        for x,y in batch:
            delta_w , delta_b = self.backprop(x,y)
            nabla_w = [ori_w+dw for ori_w, dw in zip(nabla_w, delta_w)]
            nabla_b = [ori_b+db for ori_b, db in zip(nabla_b, delta_b)]

        length = len(batch)
        self.weights = [w - learning_rate*(dw/length+regular*w) for w,dw in zip(self.weights,nabla_w)]
        self.biases = [b - (learning_rate/length)*db for b,db in zip(self.biases,nabla_b)]

    def train(self,data_train,learning_rate,epochs,batch_size,regular=0,test=None,save=None):
        """
            Main model training function.
            data_train : [[x,y],[x,y],[x,y],...,] where x is [[x_1],[x_2],...,[x_28*28]]
        """
        length = len(data_train)
        for i in range(epochs):
            random.shuffle(data_train)
            batch = [data_train[k:k+batch_size] for k in range(0,length,batch_size)]
            for each in batch:
                self.update(each,learning_rate,regular)
            print("Epoch {0}".format(i))

            if test:
                print("Test accuracy: {0}".format(self.eval(test)))
        if save:
            with open(save,'wb') as f:
                pickle.dump((self.layers,self.weights,self.biases,self.Activation),f)

    def restore(self,save):
        with open(save,'rb') as f:
            self.layers,self.weights,self.biases,self.Activation =  pickle.load(f)


    def eval(self,test):
        result = [np.argmax(self.feedforward(x))==np.argmax(y) for x,y in test]
        return sum(result)/len(result)
