'''  
Copyright (c) 2017 Intel Corporation.
Licensed under the MIT license. See LICENSE file in the project root for full license information.
'''
import time
import datetime
import os
import sys
import numpy as np
import cv2
import copy
from inference import Network
from argparse import ArgumentParser
import pathlib

#constants for frame weightage
PEOPLE_COUNT_WEIGHTAGE = 0.55
COLOR_MAP_WEIGHTAGE = .45
model_xml = ''

def build_argparser():
        parser = ArgumentParser()
        parser.add_argument("-d", "--device",	
                           help="Specify the target device to infer on; "
                                "CPU,GPU,FGPA,HDDL or MYRAID is acceptable"
                                "Application will look for a suitable plugin for device"
                                "CPU is default", default = "CPU", type=str)
        
        parser.add_argument("-m", "--model",
                        help="Path to an .xml file with a trained model.",
                        required=True, type=str)

        parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers. Absolute "
                             "path to a shared library with the kernels impl.",
                        type=str, default=None)

        return parser



def main():
    args = build_argparser().parse_args()
    cap = cv2.VideoCapture('vtest.avi')
    # pip install opencv-contrib-python
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # number of frames is a variable for development purposes, you can change the for loop to a while(cap.isOpened()) instead to go through the whole video 350
    num_frames = 1000 
    #Initialize the class
    infer_network = Network()
    #Load the network to IE plugin to get shape of input layer
    n, c, h, w = infer_network.load_model(args.model, args.device, 1, 1, 0, args.cpu_extension)[1]
   
 


    first_iteration_indicator = 1
    for i in range(0, num_frames):
	        
        '''
        There are some important reasons this if statement exists:
            -in the first run there is no previous frame, so this accounts for that
            -the first frame is saved to be used for the overlay after the accumulation has occurred
            -the height and width of the video are used to create an empty image for accumulation (accum_image)
        '''
        if (first_iteration_indicator == 1):
            ret, frame = cap.read()
            first_frame = copy.deepcopy(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape[:2]
            accum_image = np.zeros((height, width), np.uint8)
            first_iteration_indicator = 0
           
        else:
            ret, frame = cap.read()  # read a frame
            #in_frame = cv2.resize(frame, (w, h))
            # Change data layout from HWC to CHW
            #in_frame = in_frame.transpose((2, 0, 1))
            i#n_frame = in_frame.reshape((n, c, h, w))



            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale

            fgmask = fgbg.apply(gray)  # remove the background

            # for testing purposes, show the result of the background subtraction
            #cv2.imshow('diff-bkgnd-frame', fgmask)

            # apply a binary threshold only keeping pixels above thresh and setting the result to maxValue.  If you want
            # motion to be picked up more, increase the value of maxValue.  To pick up the least amount of motion over time, set maxValue = 1
            thresh = 2
            maxValue = 2
            ret, th1 = cv2.threshold(fgmask, thresh, maxValue, cv2.THRESH_BINARY)
            # for testing purposes, show the threshold image
            #cv2.imwrite('diff-th1.jpg', th1)

            # add to the accumulated image
            accum_image = cv2.add(accum_image, th1)
            # for testing purposes, show the accumulated image
            #cv2.imwrite('diff-accum.jpg', accum_image)
            cv2.imshow('diff-accum.jpg', accum_image)

            # for testing purposes, control frame by frame
             #raw_input("press any key to continue")

            # for testing purposes, show the current frame
            #cv2.imshow('frame', gray)

        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break
            #cv2.waitKey(1) & 0xFF == ord('q')

    # apply a color map
    # COLORMAP_PINK also works well, COLORMAP_BONE is acceptable if the background is dark
            color_image = im_color = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
    # for testing purposes, show the colorMap image
    #cv2.imwrite('diff-color.jpg', color_image)
    #cv2.imshow('diff-color.jpg', color_image)
   
    # overlay the color mapped image to the first frame
            result_overlay = cv2.addWeighted(first_frame, 0.7, color_image, 0.7, 0)

            # save the final overlay image
            #cv2.imwrite('diff-overlay.jpg', result_overlay)
            cv2.imshow('diff-overlay.jpg', result_overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # cleanup
            cap.release()
            cv2.destroyAllWindows()

if __name__=='__main__':
    main()
