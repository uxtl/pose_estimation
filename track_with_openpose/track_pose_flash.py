#coding:utf-8
#  post-process of pose region; change to tracking region
# input network output 
# output object region
# def get_tracking_region():
#     return x,y,w,h


# KCF track by region result
# input: region
# output: tracking result
# def KCF_tracking():


#     return x,y,w,h

import cv2 as cv 
import numpy as np
import scipy
import PIL.Image
import math
import time
from config_reader import config_reader
import util
import copy
import pylab as plt
import scipy
from scipy.ndimage.filters import gaussian_filter
caffe_path = '***'
import sys, os
sys.path.insert(0, os.path.join(caffe_path, 'python'))
import caffe

import kcftracker

def calculateIoU(candidateBound, groundTruthBound):
    cx1 = candidateBound[0]
    cy1 = candidateBound[1]
    cx2 = candidateBound[2] + cx1
    cy2 = candidateBound[3] + cy1
 
    gx1 = groundTruthBound[0]
    gy1 = groundTruthBound[1]
    gx2 = groundTruthBound[2] + gx1
    gy2 = groundTruthBound[3] + gy1
 
    carea = (cx2 - cx1) * (cy2 - cy1) 
    garea = (gx2 - gx1) * (gy2 - gy1) 
 
    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h 
    iou = area*1.0 / (carea + garea - area)
    return iou

def check_modify_vid_range(left, top, img_width, img_height, round_factor_wid, round_factor_hei, vid_width, vid_height):
    left_n = left - round_factor_wid
    top_n = top - round_factor_hei
    height_n = img_height + 2*round_factor_hei
    width_n = img_width + 2*round_factor_wid
    if(left_n+img_width > vid_width): left_n = vid_width-img_width
    if(top_n+img_height>vid_height):top_n=vid_height-img_height
    if(left_n < 0 ):
        left_n = 0
        width_n = vid_width
    if(top_n < 0): 
        top_n = 0
        height_n = vid_height
    return left_n, top_n, width_n, height_n

def initialization():
    param, model = config_reader()

    if param['use_gpu']: 
        caffe.set_mode_gpu()
        caffe.set_device(param['GPUdeviceNumber']) # set to your device!
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(model['deployFile'], model['caffemodel'], caffe.TEST)

    return net, param, model

def cal_heatmap(oriImg, net, param, model):
    #test_image = 'test.jpg'
    #test_image = '../sample_image/upper.jpg'
    #test_image = '../sample_image/upper2.jpg'
    #oriImg = cv.imread(test_image) # B,G,R order
    #heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))

    
    #multiplier = [x * model['boxsize'] / oriImg.shape[0] for x in param['scale_search']]

    scale = 1.0*model['boxsize'] / oriImg.shape[0]
    #print oriImg.shape
    #print scale
    imageToTest = cv.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
    imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model['stride'], model['padValue'])

    net.blobs['data'].reshape(*(1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))
    #net.forward() # dry run
    net.blobs['data'].data[...] = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5;
    #start_time = time.time()
    output_blobs = net.forward()
    #print('At scale %d, The CNN took %.2f ms.' % (m, 1000 * (time.time() - start_time)))

    # extract outputs, resize, and remove padding
    heatmap = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[1]].data), (1,2,0)) # output 1 is heatmaps
    heatmap = cv.resize(heatmap, (0,0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
    heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
    heatmap = cv.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)

    return heatmap


def cal_bbox(heatmap_avg, param):
    min_x, min_y = heatmap_avg.shape[:2]
    max_x = max_y = 0
    detected = False
    init_shape = heatmap_avg[:,:,0].shape
#     map_left = np.zeros(init_shape)
#     map_right = np.zeros(init_shape)
#     map_up = np.zeros(init_shape)
#     map_down = np.zeros(init_shape)
    
    labels = []
    for part in range(19-1):
        st0 = time.time()

        map_ori = heatmap_avg[:,:,part]
        #map_ori = cv.GaussianBlur(map_ori, (9, 9), 3)
      
        peak = np.unravel_index(np.argmax(map_ori, axis=None), map_ori.shape)
        #if(map_ori[peak] < 0.4): continue
        peak = (peak[1], peak[0])
        labels.append(peak)
        
        
        min_x = min(min_x, peak[0])
        min_y = min(min_y, peak[1])
        max_x = max(max_x, peak[0])
        max_y = max(max_y, peak[1])
        
        detected = True

    if detected:
        return [min_x, min_y, max_x - min_x, max_y - min_y], labels
    else:
        return [-1, -1, -1, -1],labels


# main: ntwk -> get tracking region -> KCF tracking imgae -> 
# cropped image for ntwk

if __name__ == '__main__':
    
    net, param, model = initialization()
    tracker = kcftracker.KCFTracker(True, True, True)  # hog, fixed_window, multiscale

	#if you use hog feature, there will be a short pause after you draw a first boundingbox, that is due to the use of Numba.
    marg_w = 100
    marg_h = 50
    cap = cv.VideoCapture('test.mp4')
    #cv.namedWindow('tracking')
    position_change = True
    tracker_init = False
    v_save_path = './video_res/'
    i = 0
    while(cap.isOpened()):

        ret, frame = cap.read()
        if not ret:
            break
        st0 = time.time()
        if(tracker_init):
            boundingbox = tracker.update(frame)
            boundingbox = map(int, boundingbox)

            range_x,range_y,range_w,range_h = check_modify_vid_range(boundingbox[0], boundingbox[1], boundingbox[2], boundingbox[3], marg_w, marg_h, frame.shape[1], frame.shape[0])
            frame2 = frame[range_y :range_y+range_h, range_x:range_x+range_w]

            #######################################
            print 'tracking ', time.time() - st0
            heatmap_avg = cal_heatmap(frame2, net, param, model)
            print 'mid ', time.time() - st0
            boundingbox_ntwk, sk_points = cal_bbox(heatmap_avg, param)
            print 'heatmap ', time.time() - st0
            #######################################


            if(boundingbox_ntwk[0] != -1):
                boundingbox_ntwk[0] = boundingbox_ntwk[0] + range_x
                boundingbox_ntwk[1] = boundingbox_ntwk[1] + range_y
                #print boundingbox_ntwk, boundingbox
                marg_w = boundingbox_ntwk[2]/3
                marg_h = boundingbox_ntwk[3]/6
                
                position_change = (calculateIoU(boundingbox, boundingbox_ntwk) < 0.9)

                if(position_change):
                    tracker.init(boundingbox_ntwk, frame)
                    position_change = False
                sk_pos = 0
                for p_location in sk_points:
                    #print p_location[0]+range_x, p_location[1] + range_y
                    cv.circle(frame,(p_location[0]+range_x, p_location[1]+range_y),3,(255,0,255),2)
                    font=cv.FONT_HERSHEY_SIMPLEX
                    cv.putText(frame, str(sk_pos), (p_location[0]+range_x,p_location[1]+range_y), font, 1.2,(0,255,0),1)
                    sk_pos += 1
                    
                cv.rectangle(frame,(boundingbox[0],boundingbox[1]), (boundingbox[0]+boundingbox[2],boundingbox[1]+boundingbox[3]), (0,255,255), 1)
                cv.rectangle(frame,(range_x,range_y), (range_x+range_w,range_y+range_h), (0,255,255), 1)
                cv.rectangle(frame,(boundingbox_ntwk[0],boundingbox_ntwk[1]), (boundingbox_ntwk[0]+boundingbox_ntwk[2],boundingbox_ntwk[1]+boundingbox_ntwk[3]), (255,0,255), 1)
        else:# first frame
            # frame to ntwk, post process
            #######################################
            heatmap_avg = cal_heatmap(frame, net, param, model)
            boundingbox_ntwk, sk_points = cal_bbox(heatmap_avg, param)
            #######################################
            
            if(boundingbox_ntwk[0] == -1):
                pass # need modif
            else:
                tracker.init(boundingbox_ntwk, frame)
                tracker_init = True
                marg_w = boundingbox_ntwk[2]/2
                marg_h = boundingbox_ntwk[3]/4
                #print marg_w, marg_h
            for p_location in sk_points:
                cv.circle(frame,(p_location[0], p_location[1]),3,(255,0,255),2)
            cv.rectangle(frame,(boundingbox_ntwk[0],boundingbox_ntwk[1]), (boundingbox_ntwk[0]+boundingbox_ntwk[2],boundingbox_ntwk[1]+boundingbox_ntwk[3]), (255,0,255), 1)
         
        print time.time()-st0
        cv.imwrite(v_save_path + str(i).rjust(5,'0')+'.jpg', frame)
        i = i + 1

        #duration = 0.8*duration + 0.2*(t1-t0)
        #duration = t1-t0
        #cv.putText(frame, 'FPS: '+str(1/duration)[:4].strip('.'), (8,20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        #cv.imshow('tracking', frame)
        # c = cv.waitKey(inteval) & 0xFF
        # if c==27 or c==ord('q'):
        #     break

    cap.release()
