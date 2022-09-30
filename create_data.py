import numpy
#import os
#import re
#import pickle
#import timeit
import glob
import cv2
import random
from skimage import transform
import skimage
#from skimage import io

import sklearn
from sklearn.model_selection import train_test_split   ### import sklearn tool

import keras
#from keras.preprocessing import image as image_utils
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


def load_set(videofile):
    '''The input is the path to the video file - the training videos are 99 frames long and have resolution of 720x1248
       This will be used for each video, individially, to turn the video into a sequence/stack of frames as arrays
       The shape returned (img) will be 99 (frames per video), 144 (pixels per column), 256 (pixels per row))
    '''
    ### below, the video is loaded in using VideoCapture function
    vidcap = cv2.VideoCapture(videofile)
    ### now, read in the first frame
    #success,image = vidcap.read()
    #count = 0       ### start a counter at zero
    error = ''      ### error flag
    success = True  ### start "success" flag at True
	#img = []        ### create an array to save each image as an array as its loaded 
    while success: ### while success == True
        success, img = vidcap.read()  ### if success is still true, attempt to read in next frame from vidcap video import
        #count += 1  ### increase count
        frames = []  ### frames will be the individual images and frames_resh will be the "processed" ones
        for j in range(99):
            try:
                success, img = vidcap.read()
                ### conversion from RGB to grayscale image to reduce data
                tmp = skimage.color.rgb2gray(numpy.array(img))
                ### ref for above: https://www.safaribooksonline.com/library/view/programming-computer-vision/9781449341916/ch06.html
                
                ### downsample image
                #tmp = skimage.transform.downscale_local_mean(tmp, (5,5))
                
                #this is for resizing the image
                #print(len(tmp))
                tmp = transform.resize(tmp, (144, 256))
                frames.append(tmp)
                if(len(frames)==99):
                    success=False
            
            except:
                #count+=1
                pass#print 'There are ', count, ' frame; delete last'        read_frames(videofile, name)
    
        ### if the frames are the right shape (have 99 entries), then save
        #print numpy.shape(frames), numpy.shape(all_frames)
        
        #if numpy.shape(frames)==(99, 144, 256):
        #    all_frames.append(frames)
        ### if not, pad the end with zeros
        #elif numpy.shape(frames[0])==(144,256):
            #print shape(all_frames), shape(frames), shape(concatenate((all_frames[-1][-(99-len(frames)):], frames)))
            #print numpy.shape(all_frames), numpy.shape(frames)
        #    all_frames.append(numpy.concatenate((all_frames[-1][-(99-len(frames)):], frames)))
        if numpy.shape(frames[0])!=(144,256):
            error = 'Video is not the correct resolution.'
    vidcap.release()
    return frames, error

#To load horizontally flipped frames
def hori_flipped_load_set(videofile):
        vidcap = cv2.VideoCapture(videofile)
        error = ''
        success = True
        while success: ### while success == True
            success, img = vidcap.read()  ### if success is still true, attempt to read in next frame from vidcap video import
            #count += 1  ### increase count
            frames = []  ### frames will be the individual images and frames_resh will be the "processed" ones
            for j in range(99):
                try:
                    success, img = vidcap.read()
                    ### conversion from RGB to grayscale image to reduce data
                    tmp = skimage.color.rgb2gray(numpy.array(img))
                    ### ref for above: https://www.safaribooksonline.com/library/view/programming-computer-vision/9781449341916/ch06.html
                    
                    ### downsample image
                    #tmp = skimage.transform.downscale_local_mean(tmp, (5,5))
                    
                    #this is for resizing the image
                    #print(len(tmp))
                    tmp = skimage.transform.resize(tmp, (144, 256))
                    tmp = numpy.array(tmp)
                    tmp = numpy.flip(tmp, axis = 1)
                    frames.append(tmp)
                    if(len(frames)==99):
                        success=False
                    #count+=99
                
                except:
                    #count+=1
                    pass#print 'There are ', count, ' frame; delete last'        read_frames(videofile, name)
        
            ### if the frames are the right shape (have 99 entries), then save
            #print numpy.shape(frames), numpy.shape(all_frames)
            
            #if numpy.shape(frames)==(99, 144, 256):
            #    all_frames.append(frames)
            ### if not, pad the end with zeros
            #elif numpy.shape(frames[0])==(144,256):
                #print shape(all_frames), shape(frames), shape(concatenate((all_frames[-1][-(99-len(frames)):], frames)))
                #print numpy.shape(all_frames), numpy.shape(frames)
            #    all_frames.append(numpy.concatenate((all_frames[-1][-(99-len(frames)):], frames)))
            #elif numpy.shape(frames[0])!=(144,256):
                error = 'Video is not the correct resolution.'
        vidcap.release()
        return frames, error	



def make_dataset(rand):
        seq1 = numpy.zeros((len(rand), 99, 144, 256))   ### create an empty array to take in the data
        for i,fi in enumerate(rand):                    ### for each file...
            print((i, fi))                               ### as we go through, print out each one
            if fi[-4:] == '.mp4' and i%2==0:			# even indices will be original images
                t,str = load_set(fi)                        ### load in the video file using previously defined function if .mp4 file
            #elif fi[-4:]=='.pkl':
            #    t = pickle.load(open(fi, 'rb'))
            elif fi[-4:] == '.mp4' and i%2==1:
                t,str = hori_flipped_load_set(fi)
           
            print(len(t))      ### otherwise, if it's pickled data, load the pickle
            if len(t)==(99):                  ### double check to make sure the shape is correct, and accept
                seq1[i] = t                             ### save image stack to array
            else:# TypeError:
                #'Image has shape ', shape(t), 'but needs to be shape', shape(seq1[0]) ### if exception is raised, explain
                print("Error")
                pass                                    ### continue loading data
        print((seq1.shape))
        return seq1