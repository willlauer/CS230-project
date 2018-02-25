
# coding: utf-8

# In[107]:


import tensorflow as tf
import numpy as np
from PIL import Image
import os
from tensorflow.python.framework import ops


# In[173]:


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0], 'X')
    Y = tf.placeholder(tf.float32, [None, n_y], 'Y')
    
    return X, Y


# In[230]:


# https://www.tensorflow.org/versions/r0.12/api_docs/python/image/working_with_bounding_boxes
# tf.image.draw_bounding_boxes(images, boxes, name=None)

def binaryClassificationCost(Z_final, Y):
    # I believe that sigmoid cross entropy is the one we want here, since
    # we're doing binary classification
    print(Z_final.shape, Y.shape)
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = Z_final, labels = Y)
    cost = tf.reduce_mean(cost)
    return cost
 
# The idea is to return a 7x7x2x1 vector of ious, but I'm not sure if it's vectorized correctly
# box1 is Z_fin, box2 is Y - can max function ev
def iou(box1, box2):
    xi1 = tf.maximum(box1[:, :, :, 0], box2[:, :, :, 0])
    yi1 = tf.maximum(box1[:, :, :, 1], box2[:, :, :, 1])
    xi2 = tf.minimum(box1[:, :, :, 2], box2[:, :, :, 2])
    yi2 = tf.minimum(box1[:, :, :, 3], box2[:, :, :, 3])
    inter_area = tf.maximum(tf.subtract(xi2, xi1), 0) * tf.maximum(tf.subtract(yi2, yi1), 0)
 
    box1_area = tf.subtract(box1[:, :, :, 2], box1[:, :, :, 0]) * tf.subtract(box1[:, :, :, 3], box1[:, :, :, 1])
    box2_area = tf.subtract(box2[:, :, :, 2], box2[:, :, :, 0]) * tf.subtract(box2[:, :, :, 3], box2[:, :, :, 1])
    union_area = tf.subtract(tf.add(box1_area, box2_area), inter_area)
 
    i_over_u = tf.divide(inter_area, union_area)
      
    return i_over_u

# Need to determine S and B
# Update: save this for later; just go with S = 7 and B = 2 for now.
    # Maybe split in to 10 segments, with max 1 face per? I think this is a reasonable estimate
    # of the numbers of people we'd be seeing in the images.
    # Our output should have shape SxSx(B*5*numClasses)
    # If we start only predicting 'face' or 'no face', then numClasses = 2 and we should be 
    # able to sort of disregard the output. Or we could make numClasses = 1, and only display
    # bounding boxes that have confidence past some threshold?
def detectionCost(Z_fin, Y):

    Y = tf.reshape(Y, [7, 7, 2, 5])
    Z_fin = tf.reshape(Z_fin, [7, 7, 2, 5])
    print(Z_fin.shape, Y.shape)
    # We need to break Y down into its components
    lambdCoord = 5
    lambdNoobj = 0.5
    S, B = 7, 2
    Y_confidence = Y[:, :, :, 0]
    mask = Y_confidence > 0.2
    mask = tf.layers.flatten(mask)
    
    noobj_mask = Y < 0.2;
    
    # xy loss
    xydiff = (Z_fin[:, :, :, 1] - Y[:, :, :, 1])**2 + (Z_fin[:, :, :, 2] - Y[:, :, :, 2])**2
    xydiff = tf.layers.flatten(xydiff)
    xyloss = lambdCoord * tf.reduce_sum(tf.boolean_mask(xydiff, mask))
    
    # wh loss
    whdiff = (tf.sqrt(Z_fin[:, :, :, 3]) - tf.sqrt(Y[:, :, :, 3]))**2 + (tf.sqrt(Z_fin[:, :, :, 4]) - tf.sqrt(Y[:, :, :, 4]))**2
    whdiff = tf.layers.flatten(whdiff)
    whloss = lambdCoord * tf.reduce_sum(tf.boolean_mask(whdiff, mask))  

    # confidence loss
    box_confidence = Z_fin[:, :, :, 0]
    i_over_u = iou(Z_fin, Y)
    confidence = box_confidence * i_over_u
    # 7, 7, 2
    y_hat_confidence = Y_confidence * i_over_u
    print(confidence.shape, y_hat_confidence.shape)
    
    obj_confidence = tf.boolean_mask((confidence - y_hat_confidence)**2, mask)
    noobj_confidence = lambdNoobj  * tf.boolean_mask((confidence - y_hat_confidence)**2, noobj_mask)
    
    obj_confidence = tf.reduce_sum(obj_confidence)
    noobj_confidence = tf.reduce_sum(noobj_confidence)
       
    # calculates for only 1 image right now 
    loss = xyloss + whloss + obj_confidence + noobj_confidence
    
    return loss
    
    # We don't need getCell() anymore here, right?
    #def getCell(x, y):
        # x cell is 0-indexed starting at the left
        # y cell is 0-indexed starting at the top
    #    return (x / imageWidth, y / imageHeight)

# Filters out bounding boxes below a certain threshold (0.2 for training?)
# Returns a [None] array of confidence scores and a [None, 4] array of bounding box coordinates
def filter_boxes(Z_fin):   
    threshold = 0.2
    Z_fin = tf.reshape(Z_fin, [7,7,2,5])

    # box_confidence should have a shape of 7x7x2x1, and should contain the confidence for each
    # of the generated bounding boxes
    box_confidence = Z_fin[:, :, :, 0]
    
    # boxes should have a shape of 7x7x2x4 and should contain the bounding box coordinates of each box
    boxes = Z_fin[:,:,:, 1:5]
  
    # boolean mask that filters the box_confidence tensor according to the decided threshold
    # should have shape 7x7x2x1
    mask = box_confidence > threshold
    # mask = tf.greater(box_confidence, threshold * tf.ones_like(box_confidence))
 
    # filter out confidence scores less than the threshold and reshape into a 1D tensor of scores
    confidence_scores = tf.boolean_mask(box_confidence, mask)  
    #confidence_scores = tf.multiply(original_tensor, tf.cast(mask, original_tensor.type()))
    #print(confidence_scores)
    #confidence_scores = tf.reshape(confidence_scores, confidence_scores)
    
    # filter out boxes with confidence scores less than the threshold and reshape into a [num_boxes, 4] tensor
    boxes = tf.boolean_mask(boxes, mask)
    
    #boxes = tf.reshape(boxes, (len(boxes),4))
    
    return confidence_scores, boxes
    
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):

    np.random.seed(seed)            
    m = X.shape[0]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, mini_batch_size * num_complete_minibatches : ]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * num_complete_minibatches : ]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
    
    
 


# In[231]:


def loadPretrainData():
    '''
    Loads in the binary classification images and resize them so they have
    constant dimensions h, w 
    Changes in resolution should not matter too much, but we need more exploration into
    how the aspect ratio changes impact facial detection
    
    We used the Caltech101 dataset for this task. The data is grouped into 101 categories, all but
    two of which do not contain faces. We split the data up into two directories, Class0 and Class1
    and used this split to label our data. May need more positive examples to achieve better learning
    
    Returns two (n, 448, 448, 3) tensors and two (n, 1) vectors, where n is just an integer
    '''
    class0, class1 = [], [] # Lists of tensors

    counter = 0
    # Read in all images without faces
    for subdir, dirs, files in os.walk('/home/ubuntu/Pretraining_data/Class0'):
        for f in files:
            if counter > 10: break # Memory error again
            if f.find('.DS_Store') == -1:
                img = Image.open(subdir + '/' + f)            
                img = img.resize((448, 448))
                img = np.array(img)

                #assert(img.shape == (448, 448, 3))
                if img.shape == (448, 448, 3):
                    # Only add color images
                    class0.append(img)
                    counter += 1
    counter = 0
    # Read in images with faces
    for subdir, dirs, files in os.walk('/home/ubuntu/Pretraining_data/Class1'):
        for f in files:
            if counter > 10: break # Same memory error
            if f.find('.DS_Store') == -1:
                img = Image.open(subdir + '/' + f)
                img = img.resize((448, 448))
                img = np.array(img)
                #assert(img.shape == (448, 448, 3))
                if img.shape == (448, 448, 3):
                    # only add color images
                    class1.append(img)
                    counter += 1

    print(len(class0), len(class1))
    numClass0, numClass1 = len(class0), len(class1) 
    m = numClass0 + numClass1       
    X_pretrain = np.zeros((m, 448, 448, 3))
    Y_pretrain = np.zeros((m, 1))
    
    # Label our data
    for i in range(numClass0):
        X_pretrain[i,:,:,:] = class0[i]
        Y_pretrain[i,:] = 0
    for i in range(numClass1):
        X_pretrain[i + numClass0,:,:,:] = class1[i]
        Y_pretrain[i + numClass0,:] = 1
    
    # Shuffle the results so we can get training and testing datasets
    permutation = list(np.random.permutation(m))
    shuffled_X = X_pretrain[permutation, :, :, :]
    shuffled_Y = Y_pretrain[permutation, :] 
    # Split the data into training and testing datasets
    # Not as much data in this set, so we're going with a slightly 
    # more even split of 90/10
    X_pretest = shuffled_X[:int(m/10)]
    Y_pretest = shuffled_Y[:int(m/10)]
    X_pretrain = shuffled_X[int(m/10):]
    Y_pretrain = shuffled_Y[int(m/10):]
    return X_pretrain, Y_pretrain, X_pretest, Y_pretest
   


# In[232]:




def forward_propagation(X, params, mode):
    '''
    Mode can take on values either 'classification' or 'detection'
    No parameters are needed for the avg pooling and fully connected layers 
    in 'classification', so we can branch that logic off within forwardProp
    '''
    
    # W1
    # Max pool
    # --------
    Z1 = tf.nn.conv2d(X, params['W1'], strides = [1, 2, 2, 1], padding = 'SAME') # 7x7x64
    A1 = tf.nn.relu(Z1) # Technically they used a leaky relu
    P1 = tf.nn.max_pool(A1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    
    # W2
    # Max pool
    # --------
    Z2 = tf.nn.conv2d(P1, params['W2'], strides = [1, 1, 1, 1], padding = 'SAME') # 3x3x192
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    
    # W3-6
    # Max pool
    # --------
    Z3 = tf.nn.conv2d(P2, params['W3'], strides = [1, 1, 1, 1], padding = 'SAME') # 1x1x128
    A3 = tf.nn.relu(Z3)
    Z4 = tf.nn.conv2d(A3, params['W4'], strides = [1, 1, 1, 1], padding = 'SAME') # 3x3x256
    A4 = tf.nn.relu(Z4)
    Z5 = tf.nn.conv2d(A4, params['W5'], strides = [1, 1, 1, 1], padding = 'SAME') # 1x1x256
    A5 = tf.nn.relu(Z5)
    Z6 = tf.nn.conv2d(A5, params['W6'], strides = [1, 1, 1, 1], padding = 'SAME') # 3x3x512
    A6 = tf.nn.relu(Z6)
    P3 = tf.nn.max_pool(A6, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    
    # W7-16
    # Max pool
    # --------
    Z7 = tf.nn.conv2d(P3, params['W7'], strides = [1, 1, 1, 1], padding = 'SAME') # 1x1x256
    A7 = tf.nn.relu(Z7)
    Z8 = tf.nn.conv2d(A7, params['W8'], strides = [1, 1, 1, 1], padding = 'SAME') # 3x3x512
    A8 = tf.nn.relu(Z8)
    Z9 = tf.nn.conv2d(A8, params['W9'], strides = [1, 1, 1, 1], padding = 'SAME') # 1x1x256
    A9 = tf.nn.relu(Z9)
    Z10 = tf.nn.conv2d(A9, params['W10'], strides = [1, 1, 1, 1], padding = 'SAME') # 3x3x512
    A10 = tf.nn.relu(Z10)
    Z11 = tf.nn.conv2d(A10, params['W11'], strides = [1, 1, 1, 1], padding = 'SAME') # 1x1x256
    A11 = tf.nn.relu(Z11)
    Z12 = tf.nn.conv2d(A11, params['W12'], strides = [1, 1, 1, 1], padding = 'SAME') # 3x3x512
    A12 = tf.nn.relu(Z12)
    Z13 = tf.nn.conv2d(A12, params['W13'], strides = [1, 1, 1, 1], padding = 'SAME') # 1x1x256
    A13 = tf.nn.relu(Z13)
    Z14 = tf.nn.conv2d(A13, params['W14'], strides = [1, 1, 1, 1], padding = 'SAME') # 3x3x512
    A14 = tf.nn.relu(Z14)
    Z15 = tf.nn.conv2d(A14, params['W15'], strides = [1, 1, 1, 1], padding = 'SAME') # 1x1x512
    A15 = tf.nn.relu(Z15)
    Z16 = tf.nn.conv2d(A15, params['W16'], strides = [1, 1, 1, 1], padding = 'SAME') # 3x3x1024
    A16 = tf.nn.relu(Z16)
    P4 = tf.nn.max_pool(A15, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    print(P4.shape)
    
    # W17-22
    # ------
    print('***')
    Z17 = tf.nn.conv2d(P4, params['W17'], strides = [1, 1, 1, 1], padding = 'SAME') # 1x1x256
    A17 = tf.nn.relu(Z17)
    Z18 = tf.nn.conv2d(A17, params['W18'], strides = [1, 1, 1, 1], padding = 'SAME') # 3x3x512
    A18 = tf.nn.relu(Z18)
    Z19 = tf.nn.conv2d(A18, params['W19'], strides = [1, 1, 1, 1], padding = 'SAME') # 1x1x256
    A19 = tf.nn.relu(Z19)
    Z20 = tf.nn.conv2d(A19, params['W20'], strides = [1, 1, 1, 1], padding = 'SAME') # 3x3x512
    A20 = tf.nn.relu(Z20)
    
    if mode == 'classification':
        # Branch off here to an average pooling layer and fully connected layer.
        # Use parameter 2 in fully connected layer since we're doing binary classification
        # - either an image contains a face(s) or it does not.
        # return from this branch
        C1 = tf.nn.avg_pool(A20, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        C2 = tf.contrib.layers.flatten(C1)
        C3 = tf.contrib.layers.fully_connected(C2, 1, activation_fn = None)
        return C3
        
    # Else, we are doing detection
    # Back on the main branch path for the bounding box problem
    Z21 = tf.nn.conv2d(A20, params['W21'], strides = [1, 1, 1, 1], padding = 'SAME') # 3x3x1024
    A21 = tf.nn.relu(Z21)
    Z22 = tf.nn.conv2d(A21, params['W22'], strides = [1, 2, 2, 1], padding = 'SAME') # 3x3x1024
    A22 = tf.nn.relu(Z22) 
    
    # W23-24
    # ------
    Z23 = tf.nn.conv2d(A22, params['W23'], strides = [1, 1, 1, 1], padding = 'SAME') # 3x3x1024
    A23 = tf.nn.relu(Z23)
    Z24 = tf.nn.conv2d(A23, params['W24'], strides = [1, 1, 1, 1], padding = 'SAME') # 3x3x1024
    A24 = tf.nn.relu(Z24)
    
    # FC1
    # ---
    A24 = tf.contrib.layers.flatten(A24) # 7x7x1024 => 50176
    A24 = tf.contrib.layers.fully_connected(A24, 4096, activation_fn = None)
     
    # FC2
    # --- 
    # Note: change 1470 based on however many bounding box predictions / classes we have
    # Z_fin is a flattened version of the 7x7x30 output in the paper. Can be reshaped if need be,
    # but Patrick seemed to be suggesting we can just leave it in the flattened form 
    # 7x7x10 = 490 or whatever ours is
    Z_fin = tf.contrib.layers.fully_connected(A24, 490, activation_fn = None) 
    
    # for binary classification of faces, 7x7 grid cells with 2 bounding boxes per grid cell,
    # our final output should be 7x7x10
    # each bounding box needs the confidence and bounding box coordinates
    Z_fin = tf.reshape(Z_fin, [7,7,10])
    
    # filter out predicted boxes below a certain threshold
    scores, boxes = filter_boxes(Z_fin)   
 
    # Non-max suppression
    # Need to decide what the third argument should be (max_output_size - how many bounding boxes max do we
    # want selected in the end? 20 for now?)
    # using the default overlap threshold of 0.5 - boxes are defined as overlapping if they are above this
    # threshold
    # we get back a tensor of indices of selected boxes
    #max_output_size = 20
    #max_boxes_size_tensor = tf.get_variable('max_output_size', dtype='int32')
    #tf.keras.backend.get_session().run(tf.variables_initializer([max_boxes_size_tensor]))
    max_output_size = tf.constant(20, dtype='int32')
    selected_boxes = tf.image.non_max_suppression(boxes, scores, max_output_size)
    
    # select boxes and scores generated by non-max suppression
    final_boxes = tf.gather(boxes, selected_boxes) 
    final_scores = tf.gather_nd(scores, selected_boxes)        

    #return final_scores, final_boxes
    print(Z_fin.shape)
    return Z_fin


# In[233]:




def load_dataset(bbx_gt_filename):
    '''
    Load the WIDER face dataset used for image detection. 
    Returns training and testing data and labels
    X_train has shape (n, 448, 448, 3), correspon
    '''
        
    def parameterize(bbx):
        '''
        Computes the section a bounding box is in, and update the 
        bounding box parameters to be in the 0-1 range
        Assumes image has been resized to 448x448x3 and that we have S = 7
        '''
        x, y, w, h = bbx
        
        # With box size 64, get the box coordinates for this bounding box 
        if (x >= 448 or y >= 448):
            print(x, y)
        boxNumX, boxNumY = int(x / 64), int(y / 64)
        
        # Scale x, y, w, h relative to the entire image
        # Divided by 449 to prevent against the case where one of the 
        # values is exactly 448
        bx, by = x/449.0, y/449.0
        bw, bh = w/449.0, h/449.0

        if not (bx <= 1 and by < 1 and bw < 1 and bh < 1):
            print (x, y, w, h)
            print (bx, by, bw, bh)


        # Returns the new, parameterized box (bx, by, bw, bh), and the box coordinates
        return boxNumX, boxNumY, bx, by, bw, bh
        
    def getFakeBox():
        '''
        Returns a placeholder bounding box for cells that do not have 2 faces in them
        Not being used at the moment; instead, we're currently just leaving all sections of the output
        tensor without a bounding box as 0's
        '''
        return (0, np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand()) 
    
    # Load the data for the facial detection task
    # Use WIDER face dataset, and edit the labels
    with open(bbx_gt_filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    
    img = None
    wscale, hscale = 0, 0    
    bboxes = np.zeros((7, 7, 10))
    boxCts = np.zeros((7, 7)) # Counters to ensure we don't go over 2 bbxs per cell
    
    # Store the images and corresponding bounding boxes
    X = []
    Y = []   
    counter = 0
    errors = []

    for line in content:

        # Getting a Sigkill9 everytime around image 12000, so this is to try and
        # return before hitting that.
        if counter == 10: break

        if line.find('--') != -1:
            if counter % 1000 == 0:
                print(counter, line)
            # We have reached a new image
            # If we have a previous image and bounding box set, so store that
            if img is not None:
                X += [img]
                Y += [bboxes]            
            counter += 1
            # Read in the image, set scaling variables, 
            # the current image, and reset variables for next loop
            img = Image.open('WIDER_train/images/'+line) # Reads in a color image
            w, h = img.size
            wscale, hscale = 448.0/w, 448.0/h
            img = img.resize((448, 448))
            img = np.array(img)
            assert(img.shape == (448, 448, 3))

        else:
            #TODO: getting an error from box coordinates exceeding 448

            bbx = [int(x) for x in line.split(' ')]
            if len(bbx) > 1:
                # We have an actual bounding box, not just a number of bbxs
                # We only care about the x, y, w, h so don't add the entire thing
 
                box = [bbx[0] * wscale, bbx[1] * hscale, bbx[2] * wscale, bbx[3] * hscale]
                
                # Horrible idea to do this, since I'm not sure why some of the
                # xy coordinates are going over 448.
                if not (box[0] < 448 and box[1] < 448):
                    errors.append((w, h, wscale, hscale, box))
                    if box[0] >= 448:
                        box[0] = 447
                    if box[1] >= 448:
                        box[1] = 447

                boxX, boxY, bx, by, bw, bh = parameterize(box)
                
                # Do these assertions work?
                assert (boxX < 7 and boxY < 7)
                assert (bx <= 1 and by <= 1 and bw <= 1 and bh <= 1)
                # Only store the bounding box if we have less than 2 currently for that cell
                if boxCts[boxX, boxY] == 0:
                    bboxes[boxX, boxY, 0:5] = 1, bx, by, bw, bh
                    boxCts[boxX, boxY] += 1
                elif boxCts[boxX, boxY] == 1:
                    bboxes[boxX, boxY, 5:10] = 1, bx, by, bw, bh
                    boxCts[boxX, boxY] += 1

    m = len(X)
    print('m = ', m)
    assert(len(X) == len(Y))
    
    X_train = np.zeros((m, 448, 448, 3))
    Y_train = np.zeros((m, 7, 7, 10))
    for i in range(m):
        X_train[i,:,:,:] = X[i]
        Y_train[i,:,:,:] = Y[i]
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X_train[permutation,:,:,:]
    shuffled_Y = Y_train[permutation,:,:,:] 
    
    # Break into 98% train, 2% test?  
    X_test = shuffled_X[:int(m/50),:,:,:]
    Y_test = shuffled_Y[:int(m/50),:,:,:]
    X_train = shuffled_X[int(m/50):,:,:,:]
    Y_train = shuffled_Y[int(m/50):,:,:,:]   

    print('finished loading data')
    return X_train, Y_train, X_test, Y_test     


# In[234]:


def initializeParameters():
    # Return initial weights for 24 convolutional layers
    def initializer():
        return tf.contrib.layers.xavier_initializer(seed = 0)
    
    # f, f, n_c_prev, n_c
    parameters = {
        'W1': tf.get_variable('W1', [7, 7, 3, 192], initializer = tf.contrib.layers.xavier_initializer(seed = 0)),
        'W2': tf.get_variable('W2', [3, 3, 192, 256], initializer = tf.contrib.layers.xavier_initializer(seed = 0)),
        'W3': tf.get_variable('W3', [1, 1, 256, 256], initializer = tf.contrib.layers.xavier_initializer(seed = 0)),
        'W4': tf.get_variable('W4', [3, 3, 256, 512], initializer = tf.contrib.layers.xavier_initializer(seed = 0)),
        'W5': tf.get_variable('W5', [1, 1, 512, 256], initializer = tf.contrib.layers.xavier_initializer(seed = 0)),
        'W6': tf.get_variable('W6', [3, 3, 256, 512], initializer = tf.contrib.layers.xavier_initializer(seed = 0)),
        'W7': tf.get_variable('W7', [1, 1, 512, 256], initializer = tf.contrib.layers.xavier_initializer(seed = 0)),
        'W8': tf.get_variable('W8', [3, 3, 256, 512], initializer = tf.contrib.layers.xavier_initializer(seed = 0)),
        'W9': tf.get_variable('W9', [1, 1, 512, 256], initializer = tf.contrib.layers.xavier_initializer(seed = 0)),
        'W10': tf.get_variable('W10', [3, 3, 256, 512], initializer = tf.contrib.layers.xavier_initializer(seed = 0)),
        'W11': tf.get_variable('W11', [1, 1, 512, 256], initializer = tf.contrib.layers.xavier_initializer(seed = 0)),
        'W12': tf.get_variable('W12', [3, 3, 256, 512], initializer = tf.contrib.layers.xavier_initializer(seed = 0)),
        'W13': tf.get_variable('W13', [1, 1, 512, 256], initializer = tf.contrib.layers.xavier_initializer(seed = 0)),
        'W14': tf.get_variable('W14', [3, 3, 256, 512], initializer = tf.contrib.layers.xavier_initializer(seed = 0)),
        'W15': tf.get_variable('W15', [1, 1, 512, 512], initializer = tf.contrib.layers.xavier_initializer(seed = 0)),
        'W16': tf.get_variable('W16', [3, 3, 512, 1024], initializer = tf.contrib.layers.xavier_initializer(seed = 0)),
        
        'W17': tf.get_variable('W17', [1, 1, 512, 512], initializer = tf.contrib.layers.xavier_initializer(seed = 0)),
        'W18': tf.get_variable('W18', [3, 3, 512, 1024], initializer = tf.contrib.layers.xavier_initializer(seed = 0)),
        'W19': tf.get_variable('W19', [1, 1, 1024, 512], initializer = tf.contrib.layers.xavier_initializer(seed = 0)),
        'W20': tf.get_variable('W20', [3, 3, 512, 1024], initializer = tf.contrib.layers.xavier_initializer(seed = 0)),
        'W21': tf.get_variable('W21', [3, 3, 1024, 1024], initializer = tf.contrib.layers.xavier_initializer(seed = 0)),
        'W22': tf.get_variable('W22', [3, 3, 1024, 1024], initializer = tf.contrib.layers.xavier_initializer(seed = 0)),
        'W23': tf.get_variable('W23', [3, 3, 1024, 1024], initializer = tf.contrib.layers.xavier_initializer(seed = 0)),
        'W24': tf.get_variable('W24', [3, 3, 1024, 1024], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    }
    return parameters


# In[235]:



def yolo(X_pretrain, Y_pretrain, X_pretest, Y_pretest,
         X_train, Y_train, X_test, Y_test,
         learning_rate = 0.009, num_epochs = 100, minibatch_size = 64, 
         print_cost = True):
    
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    
    mp, hp, wp, cp = X_pretrain.shape
    yp = Y_pretrain.shape[1]
    pcosts = []
        
    # Run pre-training
    Xp, Yp = create_placeholders(hp, wp, cp, yp)
    
    parameters = initializeParameters()

    Z_fin_pre = forward_propagation(Xp, parameters, 'classification')
    cost = binaryClassificationCost(Z_fin_pre, Yp)
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    print('init completed')
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            print('epoch ', epoch)
            _, c = sess.run([optimizer, cost], feed_dict = {Xp: X_pretrain, Yp: Y_pretrain})
            print('Cost after epoch %i: %f' % (epoch, c))
    
        predict_op = tf.argmax(Z_fin_pre, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Yp, 1))
        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        train_accuracy = accuracy.eval({Xp: X_pretrain, Yp: Y_pretrain})
        test_accuracy = accuracy.eval({Xp: X_pretest, Yp: Y_pretest})
        print('Train accuracy', train_accuracy)
        print('Test accuracy', test_accuracy)
        
    #ops.reset_default_graph() # Not sure about this
    
    m, nh0, nw0, nc0 = X_train.shape
    ny = Y_train.shape[1]
    print('ny = ', ny)
    costs = []
    
    X, Y = create_placeholders(nh0, nw0, nc0, ny)
    
    Z_fin = forward_propagation(X, parameters, 'detection')
    dcost = detectionCost(Z_fin, Y)
    
    optimizer2 = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(dcost)
    init = tf.global_variables_initializer() # Do we need to set this again?
    print('init2 completed')
    
    with tf.Session() as sess:
        sess.run(init)
    
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            
            # Temporary fix
            _, c = sess.run([optimizer2, dcost], 
                                         feed_dict = {X: X_train, Y: Y_train})
            '''
            Let's save mini-batching for later - not tryna debug this now, plus until we get the memory
            error worked out, the point is probably moot - I was only able to fit a few hundred examples in.
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            
            Scratch that, I'm trying it with 10 examples...
            
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                
                _, temp_cost = sess.run([optimizer2, dcost], 
                                         feed_dict = {X: minibatch_X, Y: minibatch_Y})
                minibatch_cost += temp_cost / num_minibatches
            '''    
                                                         
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, c))


# In[ ]:


##################################
######### Start here #############
##################################
X_pretrain, Y_pretrain, X_pretest, Y_pretest = loadPretrainData()
X_train, Y_train, X_test, Y_test = load_dataset('wider_face_train_bbx_gt.txt')


# In[ ]:


Y_train = np.reshape(Y_train, [Y_train.shape[0], Y_train.shape[1] * Y_train.shape[2] * Y_train.shape[3]]) # See if this helps
_, _, parameters = yolo(X_pretrain, Y_pretrain, X_pretest, Y_pretest,
         X_train, Y_train, X_test, Y_test,
         learning_rate = 0.009, num_epochs = 10, minibatch_size = 64, 
         print_cost = True) 

