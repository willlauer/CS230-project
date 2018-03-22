'''
Generates random pairings in the test set
'''
import numpy as np
import random
import os
from distutils.dir_util import copy_tree

def main():
    # path to file that will contain the pairs
    path_to_pairs_txt = '../data/testpairs.txt' 

    # path to directory with test images
    test_path = '../../datasets/lfw/test'
    
    # open pairs1.txt with writing permissions. 
    # opening with 'w' rather than 'w+' so that an error will be thrown if the file is not found
    with open(path_to_pairs_txt, 'w') as pairs: 

        for _, dirs, _ in os.walk(test_path): # get all directories (people names)
            for d in dirs: 

                    # get number of images in the directory
                    num_images = len([f for f in os.listdir(test_path + '/' + d) if not f[0] == '.'])

                    # either pair image with another person or with image of self
                    pairing = np.random.randint(0, 2)

                    # pair with image of self
                    if (pairing == 0):

                        # create a random matching with another image of same person
                        for i in range(1, min(6, num_images + 1)):
                            matching = np.random.randint(1, num_images + 1)

                            while matching == i:
                                matching = np.random.randint(1, num_images + 1)

                            pairs.write('%s %d %d\n' % (d, i, matching))

                    # pair with image of someone else
                    else:

                        for i in range(min(5, num_images)):
                            
                            image1 = np.random.randint(1, num_images + 1) # pick random image in current directoy

                            # pick a random other person and a random image in that directory
                            person = random.choice(os.listdir(test_path))
                            while (person == d):
                                person = random.choice(os.listdir(test_path))
                            num_images_2 = len([f for f in os.listdir(test_path + '/' + person) if not f[0] == '.'])
                            image2 = np.random.randint(1, num_images_2 + 1)

                            pairs.write('%s %d %s %d\n' % (d, image1, person, image2))
           
   
    print('images saved')

if __name__=='__main__':
    main()



            

