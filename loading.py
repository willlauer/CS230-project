from random import shuffle
from PIL import Image
import numpy as np
import cv2
import zipfile
import io
import pickle
import os.path
import sys
# Store bounding boxes as [class, xmin, ymin, xmax, ymax]

# Training
# WIDER_train
# wider_face_split/wider_face_train_bbx_gt.txt

# Validation
# WIDER_val
# wider_face_split/wider_face_val_bbx_gt.txt

def createBatches(filename, batchSize):
    # Returns an array of arrays, each of size batchSize, containing images and their bounding boxes 
    with open(filename) as f:
        content = f.read().splitlines()
        curImage = None
        curBoxes = []
        result = []
        for line in content:
            if line.find('--') != -1:
                if curImage != None:
                    # Store the current image and bounding boxes
                    result.append([curImage, curBoxes])
                    curImage = line # set the next image
                    curBoxes = [] # reset the boxes
                else:   
                    # Set the first image
                    curImage = line
            else:
                # We have either a number of bounding box or a bounding box itself
                # We don't care about the former, so discard if the array has only one element
                arr = line.split(' ')
                if len(arr) > 1:
                    # We have a bounding box
                    xmin, ymin = int(arr[0]), int(arr[1])
                    w, h = int(arr[2]), int(arr[3])
                    xmax, ymax = xmin + w, ymin + h
                    curBoxes.append([1, xmin, ymin, xmax, ymax])
        shuffle(result) 
        batches = [] 
        
        if batchSize == -1:
            return result # return a single batch containing all training examples
        
        else:
            numFullBatches = int(len(result) / batchSize)
            szRemaining = len(result) % batchSize 
            for i in range(numFullBatches): # add all filled batches
                batches.append(result[i * batchSize: (i+1) * batchSize])
            batches.append(result[numFullBatches * batchSize :]) # add the last elements

        return batches

def save_compressed(fh, **namedict):
    with zipfile.ZipFile(fh, mode='w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for k, v in namedict.items():
            with zf.open(k + '.npy', 'w', force_zip64=True) as buf:
                np.lib.npyio.format.write_array(buf, np.asanyarray(v), allow_pickle=True)

def my_write(filepath, data):
    # Write a single large file to a given filepath
    max_bytes = 2 ** 31 - 1
    bytes_out = pickle.dumps(data)
    print('true' if len(bytes_out) > max_bytes else 'false')
    with open(filepath, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])

def my_read(filepath):
    max_bytes = 2 ** 31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(filepath)
    with open(filepath, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return pickle.loads(bytes_in)

def zipfile_factory(f, *args, **kwargs):
    '''
    Taken from numpy source code
    Creates a ZipFile

    'file' can accept file, str, or pathlib.Path objects
    'args' and 'kwargs' are passed to zipfile.ZipFile constructor
    '''
    kwargs['allowZip64'] = True
    return zipfile.ZipFile(f, *args, **kwargs)

def create_validation_batch_file():
    validation_dataset = 'wider_face_split/wider_face_val_bbx_gt.txt'
    v_batch = createBatches(validation_dataset, -1)
    np.save('v_filenames', [v[0] for v in v_batch])
    np.save('v_boxes', [v[1] for v in v_batch])
    print('finished create_validation_batch_file')

def create_training_batch_file():
    training_dataset = 'wider_face_split/wider_face_train_bbx_gt.txt'
    t_batch = createBatches(training_dataset, -1)
    np.save('t_filenames', [t[0] for t in t_batch])
    np.save('t_boxes', [t[1] for t in t_batch])
    print('finished create_training_batch_file')

def save_validation_images():
    filenames = np.load('v_filenames.npy')
    v_images = [cv2.imread('WIDER_val/images/' + i) for i in filenames][:50]
    v_boxes = np.load('v_boxes.npy')[:50]
    np.savez('v_data2.npz', images = v_images, boxes = v_boxes)
    print('finished save_validation_images')

def save_training_images():
    filenames = np.load('t_filenames.npy')
    print(filenames[:10])
    t_images = [cv2.imread('WIDER_train/images/' + i) for i in filenames]
    print(t_images[0])
    t_boxes = np.load('t_boxes.npy')
    np.savez('t_data.npz', images = t_images, boxes = t_boxes)
    print('finished save_training_images')

def main(): 
    #create_validation_batch_file()
    create_training_batch_file()
    #save_validation_images()
    save_training_images()

if __name__=='__main__':
    main()
