# Basic Neural Network built from scratch

In this project, I built a very basic neural network from scratch. There is no deep learning package used such as TensorFlow or Caffe. The main purpose of this project is to implement the back-propagation algorithm, which is the cornerstone of the deep learning method.  

And I also want the network to handle different configs of the network easily.So we can experiment different structures of NN and explore the mystery of parameter settings.

## Dataset
We will use the classic dataset [MNIST](http://yann.lecun.com/exdb/mnist/). It contains a large database of grayscale images which is the handwritten digit. Our goal is to train a model to correctly recognize the handwritten digit (0~9).   

We will not use the original dataset but a more compatible one for Python. One can download the dataset [here](http://deeplearning.net/data/mnist/mnist.pkl.gz), which is easier to use in Python. There are totally 70,000 samples, which are 50,000 for training, 10,000 for validation and 10,000 for testing. Each sample contains a 28x28 vector and a label. The 28x28 vector represents the grayscale value of the 28x28 pixels in each image. The label is the true digit which the image represents.


## Usage

### Example

One can simply run ```bash example.sh``` to download the dataset and run a sample code.

### load.py

It contains function to load data into Python. After one downloads the [mnist.pkl.gz](http://deeplearning.net/data/mnist/mnist.pkl.gz) file in the same directory as *load.py*, call the *load_data()* function and it will return the desired train, validation and test dataset(as tuples).

```python
import load
train,val,test = load.load_data()
```
> The formation of data is modified so it is different from the structure read directly from mnist.pkl.gz. The modified is compatible to my later function. One can use *load_data(original=True)* instead to get the original data.

### activation.py

The activation functions are defined here. They are all subclasses of **Activation**. One can write his own user-defined activation function here. You just need to provide the definition and derivative of your function here.

```python
class SomeFunction(Activation):
  def result(self,x):
    # The definition of the function

  def prime(self,x):
    # The first derivative of the function
```

### network.py

The neural network class is defined here. There are several class methods provided to interact with our model.
```python
from activation import *
import network
```

#### Initialization

One can initialize the neural network providing the number of units in each layer and the activation function of each layer.    

```python
my_network = network.NeuralNetwork([28*28,100,50,10],[Tanh(),Relu(),Sigmoid()])
```
In this example, we construct a four-layer neural network. 1st layer has 28x28 units (the image contains 28x28 pixels). 2nd layer has 100 hidden units. 3rd layer has 50 hidden units. The last output layer has 10 units(represent the digit 0~9).

#### Train

After initializing the neural network and loading data, we can feed the data into the network. The function **train(data_train,learning_rate,epochs,batch_size,regular=0,test=None,save=None)** has some parameters:
1. data_train: The training data to feed in. Pass the **train** variable from **load_data()** to it.
2. learning_rate: The learning rate of this training process.
3. epochs: The epochs the model will be trained through.
4. batch_size: The number of samples in each batch.
5. regular: The parameter of the regularization term in our model. We use $L_2$ regularization here. Default:0.
6. test: Pass the validation dataset here. It will evaluate and print the classification accuracy on validation dataset after each epoch. If you don't need it, pass **None** here.
7. save: Pass the **pickle** file name you want here, to store the model in your disk. It will store the weight and biases matrices after training all the epochs. One can restore them using **restore(save)**.

```python
# After initializing the neural network and loading data:
my_network.train(train,0.01,30,20,0,val,"save.pkl")
```

#### Restore
The function helps restore the former trained model.
```python
new_network = network.NeuralNetwork([28*28,100,50,10],[Tanh(),Relu(),Sigmoid()])
new_network.restore("save.pkl")
```

#### Evaluation
The function to evaluate the accuracy of the model on given dataset.
```python
new_network.eval(test)
```

## Reference
http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/
http://neuralnetworksanddeeplearning.com/chap1.html
