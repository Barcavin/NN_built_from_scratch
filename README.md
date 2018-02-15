# Basic Neural Network built from scratch

In this project, I built a very basic neural network from scratch. There is no deep learning package used such as TensorFlow or Caffe. The main purpose of this project is to implement the back-propagation algorithm, which is the cornerstone of the deep learning method.  

And I also want the network to handle different configs of the network easily.So we can experiment different structures of NN and explore the mystery of parameter settings.

## Dataset
We will use the classic dataset [MNIST](http://yann.lecun.com/exdb/mnist/). It contains a large database of grayscale images which is the handwritten digit. Our goal is to train a model to correctly recognize the handwritten digit (0~9).   

We will not use the original dataset but a more compatible one for Python. One can download the dataset [here](http://deeplearning.net/data/mnist/mnist.pkl.gz), which is easier to use in Python. There are totally 70,000 samples, which are 50,000 for training, 10,000 for validation and 10,000 for testing. Each sample contains a 28x28 vector and a label. The 28x28 vector represents the grayscale value of the 28x28 pixels in each image. The label is the true digit which the image represents.


## Usage

### load.py

It contains function to load data into Python. After one download the [mnist.pkl.gz](http://deeplearning.net/data/mnist/mnist.pkl.gz) file, call the *load_data()* function and it will return the desired train, validation and test dataset(as tuples).

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
