# CS230-project
The basic model is located in yolo.py. That got scrapped shortly afterward, so the majority of our work was done using pre-trained models of YOLO and FaceNet, as mentioned in our citations.

The main code we used in the implementation of our pipeline is in image_crop.py, speed_comp.py, and loading.py. The first two create the datasets that on which ran FaceNet. loading.py modifies the annotations in the WIDER Face dataset and saves to the .npz format required by YOLO.

The models we used may be found at
https://github.com/davidsandberg/facenet
https://github.com/allanzelener/YAD2K
