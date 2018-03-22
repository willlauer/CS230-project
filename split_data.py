'''
Splits data into train and dev sets - 50/50
'''
import random
import os
from distutils.dir_util import copy_tree
import numpy as np

def main():
    orig_dir = '../../datasets/lfw/cropped'
    train_dir = '../../datasets/lfw/train/'
    test_dir = '../../datasets/lfw/test/'
    
    for _, dirs, _ in os.walk(orig_dir): # get all directories (people names)
        for d in dirs: 
            from_dir = orig_dir + '/' + d
            which_set = np.random.randint(0, 2)

            # determine if train/test and set path accordingly
            if (which_set == 0): 
                to_dir = train_dir + d
            else:
                to_dir = test_dir + d

            copy_tree(from_dir, to_dir)

    print('images saved')

if __name__=='__main__':
    main()



            

