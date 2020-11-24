from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d
import preprocess

import os
import tensorflow as tf
import numpy as np
import random
import math

def train(model, train_inputs, train_labels):
    pass


def get_batch(data, start_index, batch_size):
    return data[start_index : (start_index+batch_size)]

def test(model, test_inputs, test_labels):
    pass

def main():

    if len(sys.argv) != 2 or sys.argv[1] not in {"CNN+LSTM", "LSTM"}:
        print("USAGE: Sound Classification for Hazardous Environmental Sound")
        print("<Model Type>: [CNN+LSTM/LSTM]")
        exit()


    if sys.argv[1] == "CNN+LSTM":
        model = CNNLSTMModel(_,_)
    elif sys.argv[1] == "LSTM":
        #model = ReinforceWithBaseline(_,_)
        pass

    file_name = '../data/mfccs.pkl'
    get_data(file_name)

    train_inputs = _
    train_labels = _
    test_inputs = _
    test_labels = _

    for i in range(150):
        train(model, train_inputs, train_labels)
    acc = test(model, test_inputs, test_labels)
    print(acc)

    pass

if __name__ == '__main__':
    main()
