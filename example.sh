#!/bin/bash
file="mnist.pkl.gz"
if [ -f $file ]
then
    echo "Dataset Exists"
else
    wget "http://deeplearning.net/data/mnist/mnist.pkl.gz"
    echo "Download Successfully"
fi
python -i example.py