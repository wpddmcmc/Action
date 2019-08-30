# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
openpose_path= "~/openpose-master/build/python"
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(openpose_path);
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", default="../../../examples/media/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "/home/zhangyp/WorkSpace/openpose-master/models/"

# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1])-1: next_item = args[1][i+1]
    else: next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-','')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-','')
        if key not in params: params[key] = next_item

# Construct it from system arguments
# op.init_argv(args[1])
# oppython = op.OpenposePython()

try:
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    start = time.time()
    cap = cv2.VideoCapture(0)
    import torch
    import torch.nn as nn
    import numpy as np
    class Action_Net(nn.Module):
        def __init__(self):
            super(Action_Net, self).__init__()
            self.conv1 = nn.Sequential(  # input shape (1, 25, 3)
                nn.Conv2d(
                    in_channels=1,      # input height
                    out_channels=16,    # n_filters
                    kernel_size=(2,6),  # filter size
                    stride=1,           # filter movement/step
                    padding=1,          # padding=(kernel_size-1)/2 height and width don't change (3+2,25+2)
                ),  # output shape (16,5-2+1,27-6+1) = (16,4,22)
                nn.ReLU(),              # activation
                nn.MaxPool2d(kernel_size=(2,2)),  # sample in 2x1 space, output shape (16, 2,11)
            )
            self.conv2 = nn.Sequential(         # input shape (16, 11, 2)
                nn.Conv2d(
                    in_channels=16,  # input height
                    out_channels=32,  # n_filters
                    kernel_size= (1,4),  # filter size
                    stride=1,  # filter movement/step
                    padding=1,  # padding=(kernel_size-1)/2 height and width don't change (2+2,11+2)
                ), # output shape (32,  13-4+1, 4-1+1) = (32,4,10)
                nn.ReLU(),                      # activation
                nn.MaxPool2d(2),                # output shape (32, 1, 5)
            )
            self.fcon = nn.Linear( 32 * 5 * 2 , 120)
            self.fcon2 = nn.Linear(120,3) # fully connected layer, output 2 classes
            #self.softmax = nn.Softmax()

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(x.size(0), -1)
            x = self.fcon(x)
            output = self.fcon2 (x)
            #output = self.softmax(output)
            return output

    model = torch.load('model/net.pkl')
    while cap.isOpened():
        success, frame = cap.read()
        frame = cv2.resize(frame,(960,720))
        datum = op.Datum()
        datum.cvInputData = frame

        opWrapper.emplaceAndPop([datum])
        out_frame = datum.cvOutputData
        counter = 0
        if datum.poseKeypoints.shape == (1, 25, 3):
            for data in datum.poseKeypoints:
                for n in range(25):
                    for i in range(3):
                        if data[n][i] != 0:
                            counter = counter +1
        if counter > 60:
            input = torch.from_numpy(datum.poseKeypoints)
            input = input.resize(1,1, 3, 25)
            input = torch.tensor(input, dtype=torch.float32)
            result = model(input)
            prediction = torch.max(result, 1)[1].data.numpy()
            # print(prediction)
            text = ""
            if prediction == 0:
                text = "Stand"
            if prediction ==1:
                text = "Hands Up"
            if prediction == 2:
                text = "Sit"
            print(text)
            cv2.putText(out_frame,text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),2)
            
            #print("Body keypoints: \n")
            #input = torch.from_numpy(datum.poseKeypoints)
            #print(input)
        cv2.putText(out_frame,"PoseDetector By Michael.Chen",(700,700),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
        cv2.imshow("OpenPose 1.5.0 - Tutorial Python API", out_frame)
        key = cv2.waitKey(15)
        if key == 27: break

    end = time.time()
    print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
except Exception as e:
    print(e)
    sys.exit(-1)
