'''
Crops the original LFW data into thumbnails and stores in new folder
'''
from PIL import Image
import random
import os

def main():
    # number of boxes we will take in the x and y direction
    # note that the dimensions of these boxes will depend on the dimension of the image
    n_boxes_x = 2 
    n_boxes_y = 2
    
    cropped_dir = '../../datasets/lfw/cropped/'
    orig_dir = '../../datasets/lfw/raw'

    for _, dirs, _ in os.walk(orig_dir): # get all directories (people names)

        # Iterate through each subdirectory in the raw directory
        # (i.e. through each of the people)
        for d in dirs: 
            print('looking at person', d)
            for _, _, files in os.walk(orig_dir + '/' + d): 

                # create directory in cropped folder for person
                new_dir = cropped_dir + d
                os.makedirs(new_dir)

                # Keep track of how many images we have added to the current person directory
                counter = 1  

                # go through each original image in the directory
                images = [f for f in files if not f[0] == '.'] # ignore all hidden files, keep only images
                for img_name in images:
                        
                    # Get Image object from .jpg file
                    img = Image.open(orig_dir + '/' + d + '/' + img_name)
                    width, height = img.size
           
                    # boxes have width and height 1/3 of the image width and height
                    box_w = int(width / 3)
                    box_h = int(height / 3)
                          
                    for i in range(5): # create 5 random croppings per original image

                        # crop x, crop y correspond to upper left x, y coordinates
                        crop_x = random.randrange(0, width - box_w * n_boxes_x)

                        crop_y = random.randrange(0, height - box_h * n_boxes_y)

                        # move down and to the right by n_boxes_x and n_boxes_y to get the bottom right coordinates
                        crop = img.crop([crop_x, crop_y, min(width, crop_x + n_boxes_x * box_w), min(height, crop_y + n_boxes_y * box_h)])

                        # store cropped images in cropped images file
                        new_filename = d + '_' + format(counter, '04') + '.jpg'# save as 'name_xxxx.extension'
                        counter += 1

                        # save back to the current subdirectory for the person
                        crop.save(cropped_dir + d + '/' + new_filename)

    print('images saved')

if __name__=='__main__':
    main()



            

