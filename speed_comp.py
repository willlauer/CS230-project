'''
Generates test sets of cropped and full images to test the speed of the
baseline model and the trained, modified model.
'''
import numpy as np
import random
import os
from distutils.dir_util import copy_tree

def main():

    # files that will contain the pairs
    cropped_pairs = '../data/cropped_pairs.txt' 
    full_pairs = '../data/full_pairs.txt'

    # paths to directory with full/cropped images
    full_path = '../../datasets/lfw/raw'
    cropped_path = '../../datasets/lfw/cropped'

    # paths to test directorys
    full_test = '../../datasets/lfw/fulltest'
    cropped_test = '../../datasets/lfw/croppedtest'

    
    # open text files with writing permissions. 
    # opening with 'w' rather than 'w+' so that an error will be thrown if the file is not found
    with open(cropped_pairs, 'w') as cropped, open(full_pairs, 'w') as full: 

        # get 500 random pairings
        for i in range(500):
            print(i)
            # pick random directory (person)
            d = random.choice(os.listdir(full_path))

            # copy directory to test folders
            from_dir = full_path + '/' + d
            to_dir_full = full_test + '/' + d
            to_dir_cropped = cropped_test + '/' + d
            copy_tree(from_dir, to_dir_full)
            copy_tree(from_dir, to_dir_cropped)
            
            # get number of images in that full directory
            num_images_full = len([f for f in os.listdir(full_path + '/' + d) if not f[0] == '.'])

            # get number of images in corresponding cropped directory
            num_images_cropped = len([f for f in os.listdir(cropped_path + '/' + d) if not f[0] == '.'])

            # either pair image with another person or with image of self
            pairing = np.random.randint(0, 2)

            # if there is only 1 image, it must be paired with another person
            if (num_images_full < 2) : pairing = 1

            # pair with image of self
            if (pairing == 0):
                
                # create a random matching of full images 
                img1 = np.random.randint(1, num_images_full + 1)
                img2 = np.random.randint(1, num_images_full + 1)
                while img2 == img1:
                    img2 = np.random.randint(1, num_images_full + 1)
                full.write('%s %d %d\n' % (d, img1, img2))

                # create a random matching of corresponding cropped images 
                img1 = np.random.randint(1, num_images_cropped + 1)
                img2 = np.random.randint(1, num_images_cropped + 1)
                while img1 == img2:
                    img1 = np.random.randint(1, num_images_cropped + 1)
                cropped.write('%s %d %d\n' % (d, img1, img2))

            # pair with image of someone else
            else:
                # get random image from full and cropped directories
                img1_full = np.random.randint(1, num_images_full + 1) 
                img1_cropped = np.random.randint(1, num_images_cropped + 1) 

                # pick another random directory 
                d2 = random.choice(os.listdir(full_path))
                while (d == d2):
                        d2 = random.choice(os.listdir(full_path))

                # get random image from full and cropped d2 directories
                num_images_2_full = len([f for f in os.listdir(full_path + '/' + d2) if not f[0] == '.'])
                num_images_2_cropped = len([f for f in os.listdir(cropped_path + '/' + d2) if not f[0] == '.'])

                img2_full = np.random.randint(1, num_images_2_full + 1)
                img2_cropped = np.random.randint(1, num_images_2_cropped + 1)

                full.write('%s %d %s %d\n' % (d, img1_full, d2, img2_full))
                cropped.write('%s %d %s %d\n' % (d, img1_cropped, d2, img2_cropped))
   
    print('images saved')

if __name__=='__main__':
    main()



            

