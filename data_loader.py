import os, glob
import PIL
import cv2
import numpy as np


def load_xBD(img_path, label_path, disaster_lst):
    pre_img_arr = []
    post_img_arr = []
    damage_label = []

    for disaster in disaster_lst:
        pre_img_path = glob.glob(img_path + disaster + "/*_pre_*")
        post_img_path = glob.glob(img_path + disaster + "/*_post_*")
        
        pre_label_path = glob.glob(label_path + disaster + "/*_pre_*")
        post_label_path = glob.glob(label_path + disaster + "/*_post_*")
        
        pre_img_path.sort()
        post_img_path.sort()
        
        pre_label_path.sort()
        post_label_path.sort()
            
        for im_path1, im_path2, lbl_path1, lbl_path2 in zip(pre_img_path, post_img_path, pre_label_path, post_label_path):
            im1 = cv2.imread(im_path1)
            im2 = cv2.imread(im_path2)
            if np.sum(im1 == 0) == 0 and np.sum(im2 == 0) == 0:
                pre_img_arr.append(np.asarray(im1))
                post_img_arr.append(np.asarray(im2))
                
                lbl1 = cv2.imread(lbl_path1)
                lbl2 = cv2.imread(lbl_path2)
                
                label = lbl2 - lbl1
                damage_label.append(label)
    
    pre_img_arr = np.array(pre_img_arr)
    post_img_arr = np.array(post_img_arr)
    damage_label = np.array(damage_label)
    
    damage_label[damage_label == 255] = 1
    damage_label = damage_label[:,:,:,0]
    
    return pre_img_arr, post_img_arr, damage_label