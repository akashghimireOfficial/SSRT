
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



import numpy as np
from glob import glob
import cv2 as cv
import os 
from os.path import join,basename
from natsort import natsorted 
import random
import json
random.seed(1234)



import argparse

parser=argparse.ArgumentParser()

parser.add_argument("-v", "--video_idx", type=int, 
                    help="Select the Video  Index from Action Class")

parser.add_argument("-c", "--class_idx", type=int, 
                    help="Select the Class Index")


args = parser.parse_args()

paths={'saved_dire':join('datasets_features','smarthome','rgb','raw_img','CS','train'),
'saved_dire_val':join('datasets_features','smarthome','rgb','raw_img','CS','val'),
'rgb_raw':join('raw_dataset','CS','train'),
'json_path':join('datasets_features','smarthome','new_pose','CS','train')

}



seq_len=64


def get_json_dire(video_dire):
    json_name=os.path.basename(video_dire)[:-4]+'_pose3d.json'
    class_name=json_name.split('_')[0]
    json_dire=join(paths['json_path'],class_name,json_name)
    return json_dire


def get_bbox_info(json_dire): ## Getting BBOX from JSON

    with open(json_dire) as f:
        data=json.load(f)

    non_empty_frames=[]

    for frame_num,i in enumerate(data['frames']):
        
        if len(i)!=0:
            non_empty_frames.append(frame_num)

    bbox_frames=[0]*len(non_empty_frames)

    for i,frame_num in enumerate(non_empty_frames):
        pose2d=np.array(data['frames'][frame_num][0]['pose2d']).reshape(13,2)
        x=int(min(pose2d[:,0]))
        w=int(max(pose2d[:,0]))-int(min(pose2d[:,0]))
        y=int(min(pose2d[:,1]))-3
        h=int(max(pose2d[:,1]))-int(min(pose2d[:,1]))+3
        bbox=[x,y,w,h]
        
        bbox_frames[i]=bbox
    
    bbox_info=dict(zip(non_empty_frames,bbox_frames))
    bbox_info=dict([i for i in bbox_info.items() if min(i[1])>=0 ])
    
    return bbox_info


def crop_img(frame,bbox,h_req,w_req): ## Cropping Image from a video
    x,y,w,h=bbox
    
    
    if h >= h_req:
        
        diff_h=int((h-h_req)/2)
        
        y=int(y+diff_h)
        y1=y+h_req


        while y >frame.shape[0]:
            height_diff=y-frame.shape[0]
            if height_diff==1:
                y1=y1-height_diff
                y=y-height_diff
            else:
                height_diff=int(np.floor((y-frame.shape[0])/2))
                y1=y1-height_diff
                y=y-height_diff
        
        
            
        #Adjusting if crop boundry goes out of frame (y1> original height)
        while y1 >frame.shape[0]:
            height_diff=y1-frame.shape[0]
            if height_diff==1:
                y1=y1-height_diff
                y=y-height_diff
            else:
                height_diff=int(np.floor((y1-frame.shape[0])/2))
                y1=y1-height_diff
                y=y-height_diff
        

    else:
        diff_h=int((h_req-h)/2)
        y=int(y-diff_h)
        y1=y+h_req
    
        # Adjusting if y1 becomes negative
        while y <0:
            height_diff=frame.shape[0]-y1
            if height_diff==1:
                y1=y1+height_diff
                y=y+height_diff
                
            else:
                height_diff=int(np.floor((frame.shape[0]-y1)/2))
                y1=y1+height_diff
                y=y+height_diff
            
        #Adjusting if crop boundry goes out of frame (y1> original height)
        while y1 >frame.shape[0]:
            height_diff=y1-frame.shape[0]
            if height_diff==1:
                y1=y1-height_diff
                y=y-height_diff
            else:
                height_diff=int(np.floor((y1-frame.shape[0])/2))
                y1=y1-height_diff
                y=y-height_diff    

    if w >= w_req:
        
        diff_h=int((w-w_req)/2)
        
        x=w+diff_h
        x1=x+w_req

        
        while x >frame.shape[1]:
            w_diff=x-frame.shape[1]
            if w_diff==1:
                x1=x1-w_diff
                x=x-w_diff
            else:
                w_diff=int(np.floor((x-frame.shape[1])/2))
                x1=x1-w_diff
                x=x-w_diff


        
        while x1 >frame.shape[1]:
            w_diff=x1-frame.shape[1]
            if w_diff==1:
                x1=x1-w_diff
                x=x-w_diff
            else:
                w_diff=int(np.floor((x1-frame.shape[1])/2))
                x1=x1-w_diff
                x=x-w_diff



    
            #print('After  y :',y," y1 :",y1)    
    ## Adjusting width
    else:
        diff_w=(w_req-w)/2
        x=int(x-diff_w)
        #print(0)
        x1=x+w_req
        #print('x :',x," x1 :",x1)
        #Adjusting if crop boundry goes out of frame (x1> original width)
        
        while x <0:
            height_diff=frame.shape[1]-x1
            if height_diff==1:
                x1=x1+height_diff
                x=x+height_diff
                
            else:
                height_diff=int(np.floor((frame.shape[1]-x1)/2))
                x1=x1+height_diff
                x=x+height_diff
        
        
        while x1 >frame.shape[1]:
            w_diff=x1-frame.shape[1]
            if w_diff==1:
                x1=x1-w_diff
                x=x-w_diff
            else:
                w_diff=int(np.floor((x1-frame.shape[1])/2))
                x1=x1-w_diff
                x=x-w_diff
    # Cropping the 

    c_img=frame[y:y1,x:x1,:]

    # if c_img.shape != (h_req,w_req,3):
    #     print('Crop Img shape :',c_img.shape)
    return c_img


def sample_frames(video_dire,res_h,res_w):
    cap=cv.VideoCapture(video_dire)
    json_dire=get_json_dire(video_dire)
    bbox_info=get_bbox_info(json_dire)


    frames_idx=list(bbox_info.keys())
    
    total_frames =len(frames_idx)

    if total_frames ==0:
        return None

    skip_frames=total_frames//seq_len


    frames=[]
    idx=[]

    #print('skip frames :',skip_frames)

    if skip_frames==0:

        diff_frame=seq_len-total_frames
        padd_arr=np.zeros((diff_frame,res_h,res_w,3),dtype=np.uint8)

        for frame_number in frames_idx:
            cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
            sucess,frame=cap.read()
            
            bbox=bbox_info[frame_number]
            frame=crop_img(frame,bbox,res_h,res_w)
            frames.append(frame)
        
        frames=np.concatenate((frames,padd_arr))
        return np.array(frames,dtype=np.uint8)


    if skip_frames==1:
        selected_frames_idx=random.sample(range(total_frames),seq_len)
        selected_frames_idx=sorted(selected_frames_idx)
        selected_frames=[frames_idx[i] for i in selected_frames_idx]
        for frame_number in selected_frames:
            
            sucess,frame=cap.read()
            bbox=bbox_info[frame_number]
            #(bbox)
            c_frame=crop_img(frame,bbox,res_h,res_w)

            
            
            
            frames.append(c_frame)
        return np.array(frames,dtype=np.uint8)

    
    if skip_frames>1:

        selected_frames=[frames_idx[i] for i in range(0,total_frames,skip_frames)]

        if len(selected_frames) >seq_len:
            np.random.shuffle(selected_frames)
            selected_frames=selected_frames[:seq_len]
            selected_frames=sorted(selected_frames)
            for frame_number in selected_frames:
                cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
                sucess,frame=cap.read()
                bbox=bbox_info[frame_number]
                #print(frame_number,bbox)
                c_frame=crop_img(frame,bbox,res_h,res_w)
                # if c_frame.shape != (res_h,res_w,3):
                #     return frame,bbox
               

                frames.append(c_frame)
            
            return np.array(frames,dtype=np.uint8)
             
            #return idx,frames

        else:
            for frame_number in selected_frames:
                
                cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
                sucess,frame=cap.read()
                if not sucess:
                    print('okok')
                bbox=bbox_info[frame_number]
                
                frame=crop_img(frame,bbox,res_h,res_w)

                # if c_frame.shape != (400,400,3):
                #     return bbox,frame
                
            
                frames.append(frame)

            return np.array(frames,dtype=np.uint8)
           





        




# def extract_features(video_dire): ## takes the directory of the video as input 

#     frames=sample_frames(video_dire)

#     features=densenet.predict(frames)

    

#     len_features=features.shape[0]

#     if len_features==seq_len:
#         return features


#     if len_features < seq_len:

#         diff_frame=seq_len-len_features
#         padd_arr=np.zeros((diff_frame,1024),dtype=np.float32)
#         features=np.concatenate((features,padd_arr))
#         return features

    
    




def create_data(class_idx,video_idx):

    
   

    classNames=natsorted(os.listdir(paths['rgb_raw']))
    # print(classNames)
    # print('type: ',type(classNames))
    classNames=classNames[class_idx:]

    for classname in classNames:
        
        
        video_dires=natsorted(glob(join(paths['rgb_raw'],classname)+'/*.mp4'))
        random.shuffle(video_dires)

        len_class=len(video_dires)
        train_len=np.math.floor(len_class*0.9)


        video_dires=video_dires[video_idx:]


        

        for idx,video_dire in enumerate(video_dires):

            

            video_name=basename(video_dire)[:-4]

            features=sample_frames(video_dire,288,288)

            if idx <= train_len:
                target_folder=join(paths['saved_dire'],classname)

                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)

            else:
                target_folder=join(paths['saved_dire_val'],classname)

                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)


            

            target_dire=join(target_folder,'{}.npz'.format(video_name))

            #blosc_args = bp.DEFAULT_BLOSC_ARGS
            #blosc_args['clevel'] = 6

            np.savez_compressed(target_dire,features)
            #bp.pack_ndarray_file(features, target_dire)

          


            if (idx%10==0):
                print('{}: {} videos features extracted'.format(classname,idx+video_idx))

        video_idx=0


if __name__ == "__main__":

    print('Creating Dataset : ')

    create_data(args.class_idx,args.video_idx)

  

        
        

        

    

        

    