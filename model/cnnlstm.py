#CNN-LSTM Model (Mel & Yasmin)
import matplotlib.pyplot as plt
import numpy as np
import h5py
import librosa
import librosa.display
import os, sys
import time
import pandas as pd
import pickle

class Model(tf.keras.Model):
    def __init__(self):

        super(Model, self).__init__()

        self.vars = _

    def call(self, inputs, is_testing=False):


    def loss(self, logits, labels):

        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))


    def accuracy(self, logits, labels):

        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    pass


def get_batch(data, start_index, batch_size):
    return data[start_index : (start_index+batch_size)]

def test(model, test_inputs, test_labels):
    pass

def main():

    pass

if __name__ == '__main__':
    main()
