#CNN-LSTM Model (Mel & Yasmin)
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import sys

import librosa
import pickle
import random
import math

class CNNLSTMModel(tf.keras.Model):
    def __init__(self):

        super(Model, self).__init__()

        self.dropout = 0.25
        self.lstm_size = 256

        #adam optimizer
        self.optimizer = tf.keras.optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2=0.999,epsilon=1e-8)

        #initialize layers
        self.lstm1 = tf.keras.layers.LSTM(self.lstm_size, dropout=self.dropout_rate)
        self.dense1 = tf.keras.layers.Dense(10, activation='softmax')

        self.conv1 = tf.keras.layers.Conv2D(filters=4, kernel_size=(5,5), strides=(4,1), activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(2,1), activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(2,2), strides=(2,1), activation='relu')
        #in the paper, strides is listed as (N/A, 1)
        self.conv4 = tf.keras.layers.Conv2D(filters=300, kernel_size=(2,2), strides=(0,1), activation='relu')

        self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 1), strides=(2,1))
        self.maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 1), strides=(2,1))

        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(self.dropout_rate)


    def call(self, inputs):
        #padding='same' if this doesn't work
        convlayer1 = self.conv1(inputs)
        convlayer2 = self.conv2(convlayer1)
        maxpool1 = self.maxpool1(convlayer2)

        drop1 = self.dropout1(maxpool1)
        convlayer3 = self.conv3(drop1)
        maxpool2 = self.maxpool2(convlayer3)

        drop2 = self.dropout2(maxpool2)
        convlayer4 = self.conv4(drop2)
        drop3 = self.dropout3(convlayer4)

        reshape = tf.reshape(drop3, (300,32))
        lstm = self.lstm1(reshape)
        dense = self.dense(lstm)

        return dense

    def loss(self, logits, labels):

        '''As cited in the paper, Table IV Experimental Results, authors used Categorical
         cross entropy loss to measure their CNN+LSTM models'''
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))

    def accuracy(self, logits, labels):

        #check
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

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


    pass

if __name__ == '__main__':
    main()
