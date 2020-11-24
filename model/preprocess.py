#place to shuffle inputs, create train & test input label
import numpy as np
import sys
from random import seed, shuffle
import sdf_iterator


def read_file(file_name):
    '''Reads in pkl file with mfcc.pkl feature data from UrbanSound8K dataset'''
    with open(file_name, 'rb') as f:
        extract = pickle.load(f)
    return extract

def get_data(file_name):
    data = read_file(file_name)

    #?

    pass
