import network
import load
from activation import *

train,val,test = load.load_data()

my_network = network.NeuralNetwork([28*28,100,50,10],[Relu(),Sigmoid(),Tanh()])
my_network.train(train,0.01,30,20,0,val,"save.pkl")

print("Test accuracy:{0}".format(my_network.eval(test)))

new_network = network.NeuralNetwork([28*28,100,50,10],[Relu(),Sigmoid(),Tanh()])
new_network.restore("save.pkl")
