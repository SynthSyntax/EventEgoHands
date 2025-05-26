import numpy as np
import torch
import hdf5plugin
import h5py
import os
from pathlib import Path
from typing import Optional, Callable
from torch.utils.data import Dataset
from torch_geometric.data import Data
from dagr.data.augment import init_transforms
from dagr.data.utils import to_data
import cv2
import json
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.profile import get_data_size
import sys
import pybboxes as pbx

torch.cuda.empty_cache()


def tracks_to_array(tracks):
    return np.stack([tracks['x'], tracks['y'], tracks['w'], tracks['h'], tracks['class_id']], axis=1)

def tracks_list_to_dict(tracks):
    ret_tracks = [{'x':tracks[0],'y':tracks[1],'w':tracks[2],'h':tracks[3],'class_id':1}]
    return ret_tracks


    
class Egohands(Dataset):

    #note that everything is zero based indexing - no exceptions

    def __init__(self, root:Path, bbox_path:Path, num_events: int=50000,num_frames=2700):
        super().__init__()
        self.num_events = num_events
        self.num_frames = num_frames
        self.root = root
        self.bbox_path = bbox_path
        self.time_window = 1000000
        self.classes = ("hand")
        self.height = 720
        self.width = 1280
        self.num_classes = 1
        #assert frames in dataset to ensure completeness
        # self.assert_frames()
        # assert root.exists()
        
        with open(self.bbox_path, "r") as f:
            self.tracks_data = json.load(f)

 



    def __len__(self):
        len = (self.num_frames-1)*(self.get_num_files(self.root,"events"))
        return len


    def __getitem__(self, idx):


        
        # Retrieve the frames, events, and tracks
        image0, image1, vid_name_frames = self.get_frames(idx)

        image0, image1, vid_name_frames = self.get_frames(idx)

        if image0 is None:
            print("image0 is none here")
            image0,image1,vid_name_frames = self.get_frames(12345)
        
        if image0 is None:
            print("image1 is none here")
            image0,image1,vid_name_frames = self.get_frames(12345)

        



        image0 = self.preprocess_frames(image0)
        # image1 = self.preprocess_frames(image1)
        image0 = image0.to(torch.uint8)
        # image1 = image1.to(torch.uint8)
        events, vid_name_events = self.get_events(idx)
        tracks0, tracks1 = self.get_tracks_COCO(idx, self.bbox_path, vid_name_frames,self.tracks_data)





#new world - tracks error

        # tracks0 = np.array([[532.63403, 413.06967, 150.65317, 111.27258,   1.     ],
        #                     [671.9156,  430.4161,  159.77002, 108.9798,    1.     ]])
        
        # tracks1 = np.array([[533.05615, 412.4619,  150.51193, 111.91812,   1.     ],
        #                     [673.6106,  429.9334,  157.63733, 107.76932 ,  1.     ]])


    #     tracks0 = np.array([[225., 133., 132., 127.,   1.],
    # [713., 170., 131., 117.,   1.],
    # [  0., 615., 264., 105.,   1.],
    # [137., 363., 261., 352.,   1.]])
    

    #     tracks1 = np.array([[714., 170., 130., 118.,   1.],
    # [225., 133., 133., 127.,   1.],
    # [  0., 614. ,267., 106.,   1.],
    # [398., 650., 283.,  69.,   1.]])
    
     
        tracks0 = tracks0.astype(np.float32)
        tracks1 = tracks1.astype(np.float32)
        # print("shape of the track0", tracks0.shape)
        # print("shape of the track1", tracks1.shape)
        print("tracks0 is",tracks0)
        print("tracks1 is",tracks1)

        # # Skip invalid tracks
        # if tracks0 is None or tracks1 is None:
        #     return None
        
        # Get timestamps
        rel_idx = self.get_rel_idx(idx)
        timestamps_rgb = [int(1e6 * ((1 / 30) * i)) for i in range(self.num_frames)]
        image_ts_0, image_ts_1 = timestamps_rgb[rel_idx], timestamps_rgb[rel_idx + 1]
        # print("bbox type", tracks0.dtype)
        # Convert to torch geometric data


        # print("the frame name is", vid_name_frames)
        data = to_data(
            **events,
            bbox=tracks1,
            bbox0=tracks0,
            image=image0,
            width=self.width,
            height=self.height,
            sequence=vid_name_frames,
            time_window=self.time_window,
            t0=image_ts_0,
            t1=image_ts_1

        )
        # print("bbox type after data", data.bbox.dtype)
        print("the data is",data)
        return data



        
    def assert_frames(self):
        data_dir = self.root
        root_dir = os.path.join(data_dir, "frames")

        # Iterate through each video folder
        for vid_folder in os.listdir(root_dir):
            vid_path = os.path.join(root_dir, vid_folder)

        # Ensure it's a directory
            if os.path.isdir(vid_path):
                # print(f"Checking {vid_folder}...")
                num_images = len([f for f in os.listdir(vid_path) if os.path.isfile(os.path.join(vid_path, f))])

            # Assert condition
            assert num_images == 2700, f"Error: {vid_folder} has {num_images} images instead of 2700"

        print("All video folders have exactly 2700 images.")

        

        #use the vid name corresponding to that idx (create a number mapping for videos?) - call the get_events, call the get_tracks
        #Put all the called stuff into a dictionary to return



    def get_events(self, idx):
        #returns a dictionary of events in that timeframe
        vid_no = self.get_vid_no(idx)
        events_root = os.path.join(self.root,"events")
        rel_idx = self.get_rel_idx(idx)
        file_paths1 = []
        for file in os.listdir(events_root):  # Iterate over all items in the 'frames' folder
            file_path = os.path.join(events_root, file)  # Get the full path of the item
            file_paths1.append(file_path)


        if rel_idx>2690:
            return self.get_events(370)

        
        # print(file_paths1)
        file_paths = sorted(file_paths1)
        # print(file_paths)
        file = file_paths[vid_no]
        with h5py.File(str(file)) as fh:
            events = fh["events"]
            timeidx = fh["timeidx"]
            x = events["x"]
            y = events["y"]
            t = events["t"]
            p = events["p"]
            # if(rel_idx==2693):
            #     print(f"we're at the exception where the idx is {idx} and rel_idx is {rel_idx}")
            timestamp1 = timeidx[rel_idx]
            # timestamp2 = timeidx[rel_idx+1]
            if ((rel_idx+2)<len(timeidx)):
                timestamp2 = timeidx[rel_idx+1]
            else:
                
                # events={
                #     'p':np.array([0]),
                #     't':np.array([0]),
                #     'x':np.array([0]),
                #     'y':np.array([0]),
                # }
                return self.get_events(370)
                # vid_name = os.path.basename(file)
                # return events, vid_name

            

            # x_new = x[50:200]
            # p_new = p[50:200]
            # t_new = t[50:200]
            # y_new = y[50:200]
            x_new = x[timestamp1:(timestamp2-1)].astype(np.uint16)
            p_new = p[timestamp1:(timestamp2-1)]
            p_new = p_new.astype(np.int8)

            # print("pnew before",p_new)
            # p_new[p_new == 0] = -1 
            # print("pnew after",p_new)
            p_new = 2 * p_new.reshape((-1,1)) - 1
            # print("p_new after weird reshape",p_new)
            t_new = t[timestamp1:(timestamp2-1)].astype(np.int64)
            y_new = y[timestamp1:(timestamp2-1)].astype(np.uint16)
            # print(f"p_new is {p_new} and the datatype is : {type(p_new)}")
            # print("the length of P_new is", len(p_new))
            # print("this is the type of the event list", type(x_new))
            print("the length of x_new is",len(x_new))
            print("the length of p_new is",len(p_new))
            print("the length of t_new is",len(t_new))
            print("the length of y_new is",len(y_new))



            #outlier handling case
            if(len(x_new)<300):
                # x_new = np.random.rand(20000).astype(np.uint16)
                # y_new = np.random.rand(20000).astype(np.uint16)
                # p_new = np.random.rand(20000).astype(np.int8)
                # t_new = np.random.rand(20000).astype(np.int64)
                print("the index where this shit is happening",idx)
                print("the relative index of this shit",rel_idx)
                return self.get_events(370)
            
            
            



            
            events = {
                'p': p_new,
                't': t_new,
                'x': x_new,
                'y': y_new,
            }

        vid_name = os.path.basename(file)
        return events,vid_name
        

        


    def get_rel_idx(self, idx):
        vid_no = self.get_vid_no(idx)
        assert idx>=0, "invalid index"
        assert vid_no<self.get_num_files(self.root,"events"), "no video exists at that index"
        start_idx = (vid_no)*(self.num_frames-1)
        end_idx = start_idx + (self.num_frames-2)
        rel_idx = idx-start_idx
        return rel_idx

    def get_num_files(self, root, mode:str):
        path = os.path.join(root,mode)
        file_count = len([file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))])
        return file_count


    def get_vid_no(self,idx): #the video numbers are also zero indexing
        return (idx//(self.num_frames-1))




    def get_tracks_COCO(self, idx, bbox_path, vid_name: str,bbox_data):
        frame_idx = self.get_rel_idx(idx)
        print("the index is",idx)
        print("the relative index is", frame_idx)
        frame_key0 = f"{vid_name}_frame_{frame_idx}"
        frame_key1 = f"{vid_name}_frame_{frame_idx+1}"

        # with open(bbox_path) as f:
        #     bbox_data = json.load(f)

        # Initialize empty lists to hold the bounding boxes
        bboxes0 = []
        bboxes1 = []

        # Check if frame_key0 exists in the bounding box data
        if frame_key0 in bbox_data:
            bboxes0 = bbox_data[frame_key0]

        # Check if frame_key1 exists in the bounding box data
        if frame_key1 in bbox_data:
            bboxes1 = bbox_data[frame_key1]

        # If no bounding boxes are found, return an empty array with shape (1, 5)
        if not bboxes0 or not bboxes1:
            print("a bounding box was  not found")
            array = np.ones((1, 5))
            array[0, -1] = 1.0  # Set the class ID to 1.0
            return array, array
        
        # bboxes0=np.round(bboxes0)
        # bboxes1=np.round(bboxes1)

        

        # Convert the bounding boxes to numpy arrays
        bboxes0 = np.array(bboxes0,dtype=np.float32)
        bboxes1 = np.array(bboxes1,dtype=np.float32)
        # print("tracks before0", bbox0)
        # print("tracks before1", bbox1)

        bbox0 = np.zeros_like(bboxes0)
        bbox1 = np.zeros_like(bboxes1)


#yolo normalisation
        bbox0[:, 0] = bboxes0[:, 0] / self.width  # x_center
        # print("after the first step",bbox0)
        bbox0[:, 1] = bboxes0[:, 1] / self.height  # y_center
        bbox0[:, 2] = bboxes0[:, 2] / self.width  # width
        bbox0[:, 3] = bboxes0[:, 3] / self.height  # height


        bbox1[:, 0] = bboxes1[:, 0] / self.width  # x_center
        bbox1[:, 1] = bboxes1[:, 1] / self.height  # y_center
        bbox1[:, 2] = bboxes1[:, 2] / self.width  # width
        bbox1[:, 3] = bboxes1[:, 3] / self.height  # height


        # print("before normal full",bboxes0)
        # print("before normal full",bboxes1)

        # print("after normal", bbox0)
        # print("after bnormal",bbox1)

        #fine till here
        bbox0_new = []
        bbox1_new = []

        default = [137., 363., 261., 352.,1 ]
        for i in bbox0:
            try:
                print(i)
                x = pbx.convert_bbox(i,from_type="yolo",to_type="coco",image_size=(1280,720),strict=True)
                lst = list(x)  # Convert tuple to list
                lst.append(1)
                bbox0_new.append(lst)
                # bbox0_new = np.hstack([bbox0_new, np.ones((bbox0_new.shape[0], 1),dtype=np.float32)])


            except Exception as e:
                print(f"the error converting the bounding box occured here {e}")
                bbox0_new.append(default)



        for i in bbox1:
            try:
                print(i)
                x = pbx.convert_bbox(i,from_type="yolo",to_type="coco",image_size=(1280,720),strict=True)
                lst = list(x)  # Convert tuple to list
                lst.append(1)
                bbox1_new.append(lst)
                # bbox1_new = np.hstack([bbox1_new, np.ones((bbox1_new.shape[0], 1),dtype=np.float32)])

            except Exception as e:
                print(f"the error converting the bounding box occured here {e}")
                bbox1_new.append(default)

        bbox0_new = np.array(bbox0_new)
        bbox1_new = np.array(bbox1_new)


        if bbox0_new.size == 0:
            bbox0_new = np.array([137., 363., 261., 352., 1])

        if bbox1_new.size == 0:
            bbox1_new = np.array([137., 363., 261., 352., 1])

        print("before stacking0",bbox0_new)
        print("before stacking1",bbox1_new)

        # bbox0_new = np.hstack([bbox0_new, np.ones((bbox0_new.shape[0], 1),dtype=np.float32)])
        # bbox1_new = np.hstack([bbox1_new, np.ones((bbox1_new.shape[0], 1),dtype=np.float32)])



                

        print("bbox0 after filtering",bbox0_new)


        return bbox0_new, bbox1_new
    




    def get_tracks_corner(self, idx, bbox_path, vid_name: str,bbox_data):
        frame_idx = self.get_rel_idx(idx)
        print("the index is",idx)
        print("the relative index is", frame_idx)
        frame_key0 = f"{vid_name}_frame_{frame_idx}"
        frame_key1 = f"{vid_name}_frame_{frame_idx+1}"

        # with open(bbox_path) as f:
        #     bbox_data = json.load(f)

        # Initialize empty lists to hold the bounding boxes
        bboxes0 = []
        bboxes1 = []

        # Check if frame_key0 exists in the bounding box data
        if frame_key0 in bbox_data:
            bboxes0 = bbox_data[frame_key0]

        # Check if frame_key1 exists in the bounding box data
        if frame_key1 in bbox_data:
            bboxes1 = bbox_data[frame_key1]

        # If no bounding boxes are found, return an empty array with shape (1, 5)
        if not bboxes0 or not bboxes1:
            print("a bounding box was  not found")
            array = np.zeros((1, 5))
            array[0, -1] = 1.0  # Set the class ID to 1.0
            return array, array
        
        bboxes0=np.round(bboxes0)
        bboxes1=np.round(bboxes1)

        # Convert the bounding boxes to numpy arrays
        bbox0 = np.array(bboxes0,dtype=np.float32)
        bbox1 = np.array(bboxes1,dtype=np.float32)
        # print("tracks before0", bbox0)
        # print("tracks before1", bbox1)

        # Convert from (x_center, y_center, w, h, class) to (x_top_left, y_top_left, w, h, class)
        bbox0[:, 0] = bbox0[:, 0] - (bbox0[:, 2] / 2)  # x_top_left = x_center - w / 2
        bbox0[:, 1] = bbox0[:, 1] - (bbox0[:, 3] / 2)  # y_top_left = y_center - h / 2

        bbox1[:, 0] = bbox1[:, 0] - (bbox1[:, 2] / 2)  # x_top_left = x_center - w / 2
        bbox1[:, 1] = bbox1[:, 1] - (bbox1[:, 3] / 2) # y_top_left = y_center - h / 2

        # Add a column of ones to represent the class ID (1.0) for each bounding box
        bbox0 = np.hstack([bbox0, np.ones((bbox0.shape[0], 1),dtype=np.float32)])
        bbox1 = np.hstack([bbox1, np.ones((bbox1.shape[0], 1),dtype=np.float32)])
        # print("tracks after0", bbox0)
        # print("tracks after1", bbox1)
        bbox0=np.round(bbox0)
        bbox1=np.round(bbox1)
        print("hi")
        bbox0 = self.filter_bbox(bbox0)
        bbox1 = self.filter_bbox(bbox1)
        print("it is filtered")
        print("bbox0 after filtering",bbox0)
        print("bbox1 after filtering",bbox1)
        return bbox0, bbox1
    





    # def get_tracks_VOC(self, idx, bbox_path, vid_name: str,bbox_data):
    #     frame_idx = self.get_rel_idx(idx)
    #     print("the index is",idx)
    #     print("the relative index is", frame_idx)
    #     frame_key0 = f"{vid_name}_frame_{frame_idx}"
    #     frame_key1 = f"{vid_name}_frame_{frame_idx+1}"

    #     # with open(bbox_path) as f:
    #     #     bbox_data = json.load(f)

    #     # Initialize empty lists to hold the bounding boxes
    #     bboxes0 = []
    #     bboxes1 = []

    #     # Check if frame_key0 exists in the bounding box data
    #     if frame_key0 in bbox_data:
    #         bboxes0 = bbox_data[frame_key0]

    #     # Check if frame_key1 exists in the bounding box data
    #     if frame_key1 in bbox_data:
    #         bboxes1 = bbox_data[frame_key1]

    #     # If no bounding boxes are found, return an empty array with shape (1, 5)
    #     if not bboxes0 or not bboxes1:
    #         print("a bounding box was  not found")
    #         array = np.ones((1, 5))
    #         array[0, -1] = 1.0  # Set the class ID to 1.0
    #         return array, array
        
    #     bboxes0=np.round(bboxes0)
    #     bboxes1=np.round(bboxes1)

    #     # Convert the bounding boxes to numpy arrays
    #     bboxes0 = np.array(bboxes0,dtype=np.float32)
    #     bboxes1 = np.array(bboxes1,dtype=np.float32)
    #     # print("tracks before0", bbox0)
    #     # print("tracks before1", bbox1)

    #     # Convert from (x_center, y_center, w, h, class) to (xmin,ymin, xmax, ymax)
    #     bbox0[:, 0] = bboxes0[:, 0] - (bboxes0[:, 2] / 2)  
    #     bbox0[:, 1] = bboxes0[:, 1] + (bboxes0[:, 3] / 2)  
    #     bbox0[:, 2] = bboxes0[:, 0] + (bboxes0[:, 2] / 2)
    #     bbox0[:, 3] = bboxes0[:, 1] - (bboxes0[:, 3] / 2)

    #     bbox1[:, 0] = bboxes1[:, 0] - (bboxes1[:, 2] / 2)  
    #     bbox1[:, 1] = bboxes1[:, 1] + (bboxes1[:, 3] / 2)  
    #     bbox1[:, 2] = bboxes1[:, 0] + (bboxes1[:, 2] / 2)
    #     bbox1[:, 3] = bboxes1[:, 1] - (bboxes1[:, 3] / 2)

    #     # Add a column of ones to represent the class ID (1.0) for each bounding box
    #     bbox0 = np.hstack([bbox0, np.ones((bbox0.shape[0], 1),dtype=np.float32)])
    #     bbox1 = np.hstack([bbox1, np.ones((bbox1.shape[0], 1),dtype=np.float32)])
    #     # print("tracks after0", bbox0)
    #     # print("tracks after1", bbox1)
    #     bbox0=np.round(bbox0)
    #     bbox1=np.round(bbox1)
    #     print("hi")
    #     bbox0 = self.filter_bbox_VOC(bbox0)
    #     bbox1 = self.filter_bbox_VOC(bbox1)
    #     print("it is filtered")
    #     print("bbox0 after filtering",bbox0)
    #     print("bbox1 after filtering",bbox1)
    #     return bbox0, bbox1

    def get_tracks_corner(self, idx, bbox_path, vid_name: str,bbox_data):
        frame_idx = self.get_rel_idx(idx)
        print("the index is",idx)
        print("the relative index is", frame_idx)
        frame_key0 = f"{vid_name}_frame_{frame_idx}"
        frame_key1 = f"{vid_name}_frame_{frame_idx+1}"

        # with open(bbox_path) as f:
        #     bbox_data = json.load(f)

        # Initialize empty lists to hold the bounding boxes
        bboxes0 = []
        bboxes1 = []

        # Check if frame_key0 exists in the bounding box data
        if frame_key0 in bbox_data:
            bboxes0 = bbox_data[frame_key0]

        # Check if frame_key1 exists in the bounding box data
        if frame_key1 in bbox_data:
            bboxes1 = bbox_data[frame_key1]

        # If no bounding boxes are found, return an empty array with shape (1, 5)
        if not bboxes0 or not bboxes1:
            print("a bounding box was  not found")
            array = np.zeros((1, 5))
            array[0, -1] = 1.0  # Set the class ID to 1.0
            return array, array
        
        bboxes0=np.round(bboxes0)
        bboxes1=np.round(bboxes1)

        # Convert the bounding boxes to numpy arrays
        bbox0 = np.array(bboxes0,dtype=np.float32)
        bbox1 = np.array(bboxes1,dtype=np.float32)
        # print("tracks before0", bbox0)
        # print("tracks before1", bbox1)

        # Convert from (x_center, y_center, w, h, class) to (x_top_left, y_top_left, w, h, class)
        bbox0[:, 0] = bbox0[:, 0] - (bbox0[:, 2] / 2)  # x_top_left = x_center - w / 2
        bbox0[:, 1] = bbox0[:, 1] - (bbox0[:, 3] / 2)  # y_top_left = y_center - h / 2

        bbox1[:, 0] = bbox1[:, 0] - (bbox1[:, 2] / 2)  # x_top_left = x_center - w / 2
        bbox1[:, 1] = bbox1[:, 1] - (bbox1[:, 3] / 2) # y_top_left = y_center - h / 2

        # Add a column of ones to represent the class ID (1.0) for each bounding box
        bbox0 = np.hstack([bbox0, np.ones((bbox0.shape[0], 1),dtype=np.float32)])
        bbox1 = np.hstack([bbox1, np.ones((bbox1.shape[0], 1),dtype=np.float32)])
        # print("tracks after0", bbox0)
        # print("tracks after1", bbox1)
        bbox0=np.round(bbox0)
        bbox1=np.round(bbox1)
        print("hi")
        bbox0 = self.filter_bbox(bbox0)
        bbox1 = self.filter_bbox(bbox1)
        print("it is filtered")
        print("bbox0 after filtering",bbox0)
        print("bbox1 after filtering",bbox1)
        return bbox0, bbox1
    
    def filter_bbox(self,bbox):
        # print("inside filtering")

        filtered_bbox = []

        for i in bbox:
            #checking if the x,y boxes are within the image
            if (i[0]>0 and i[0]<self.width and i[1]>0 and i[1]<self.height and (i[0]+i[2]<self.width) and (i[1]-i[3]>0)):
                filtered_bbox.append(i)

        filtered_bbox = np.array(filtered_bbox)

        if filtered_bbox.shape[0] == 0:
            print("==========================================================there was an empty bbox here============================================================")
            filtered_bbox = np.array([[1, 1, 5, 5,   1.]])

        
        return np.array(filtered_bbox)
    

    def filter_bbox_VOC(self,bbox):
    # print("inside filtering")

        filtered_bbox = []

        for i in bbox:
            #checking if the x,y boxes are within the image
            if (i[0]>0 and i[0]<self.width and i[2]>0 and i[2]<self.width and i[1]>0 and i[1]<self.height and i[3]>0 and i[3]<self.height):
                filtered_bbox.append(i)

        filtered_bbox = np.array(filtered_bbox)

        if filtered_bbox.shape[0] == 0:
            print("==========================================================there was an empty bbox here============================================================")
            filtered_bbox = np.array([[588., 326., 293., 252.,   1.]])

        
        return np.array(filtered_bbox)

        



    def get_tracks(self, idx, bbox_path, vid_name: str):
        frame_idx = self.get_rel_idx(idx)
        print("the relative index is", frame_idx)
        frame_key0 = f"{vid_name}_frame_{frame_idx}"
        frame_key1 = f"{vid_name}_frame_{frame_idx+1}"

        with open(bbox_path) as f:
            bbox_data = json.load(f)

        # Initialize empty lists to hold the bounding boxes
        bboxes0 = []
        bboxes1 = []

        # Check if frame_key0 exists in the bounding box data
        if frame_key0 in bbox_data:
            bboxes0 = bbox_data[frame_key0]

        # Check if frame_key1 exists in the bounding box data
        if frame_key1 in bbox_data:
            bboxes1 = bbox_data[frame_key1]

        # If no bounding boxes are found, return an empty array with shape (1, 5)
        if not bboxes0 or not bboxes1:
            array = np.zeros((1, 5), dtype=np.float32)
            array[0, -1] = 1.0  # Set the class ID to 1.0
            return array, array

        # Convert the bounding boxes to numpy arrays
        bbox0 = np.array(bboxes0, dtype=np.float32)
        bbox1 = np.array(bboxes1, dtype=np.float32)

        # Add a column of ones to represent the class ID (1.0) for each bounding box
        bbox0 = np.hstack([bbox0, np.ones((bbox0.shape[0], 1), dtype=np.float32)])
        bbox1 = np.hstack([bbox1, np.ones((bbox1.shape[0], 1), dtype=np.float32)])

        return bbox0, bbox1

    


    def get_frames(self,idx): #frames/image
        vid_no = self.get_vid_no(idx)
        # path1 = os.path.join(root,"events")
        frames_root = os.path.join(self.root,"frames")
        folder_paths1 = []  # Initialize an empty list to store folder paths
        for folder in os.listdir(frames_root):  # Iterate over all items in the 'frames' folder
            folder_path = os.path.join(frames_root, folder)  # Get the full path of the item
            folder_paths1.append(folder_path)
        # print(folder_paths1)
        folder_paths=sorted(folder_paths1)
        # print(folder_paths)
        frames_path = folder_paths[vid_no]
        rel_idx = self.get_rel_idx(idx)
        frame_path1 = os.path.join(frames_path,f"frame_{rel_idx}.jpg")
        frame_path2 = os.path.join(frames_path,f"frame_{rel_idx+1}.jpg")
        vid_name = os.path.basename(frames_path)
        frame1 = cv2.imread(frame_path1)
        frame2 = cv2.imread(frame_path2)



        return frame1,frame2, vid_name
    
    def preprocess_frames(self,image):

        #These are the edits i made - which need to be reverted
        ###################################################################
        # image = image[:2 * 215]
        # image = cv2.resize(image, (320, 215), interpolation=cv2.INTER_CUBIC)
        ###################################################################

        image = torch.from_numpy(image).permute(2, 0, 1)
        image = image.unsqueeze(0)
        # print("shape of the image", image.shape)
        return image



# events = get_events_in_time_window("/cfs/earth/scratch/kotabha1/v2e/cards_courtyard_B_T_split.h5")
# print(events) 

#calls and tests
# path = "/cfs/earth/scratch/kotabha1/egohands/test/events/chess_livingroom_S_B.h5"
root = "/cfs/earth/scratch/kotabha1/egohands/train"
bbox_path = "/cfs/earth/scratch/kotabha1/egohands/bounding_boxes.json"
root_test = "/cfs/earth/scratch/kotabha1/egohands/test"
root_train = "/cfs/earth/scratch/kotabha1/egohands/train"

train_dataset = Egohands(root_train, bbox_path)
obj = Egohands(root,bbox_path)
# train_loader = DataLoader(obj, follow_batch=['bbox', 'bbox0'], batch_size=2, shuffle=False, num_workers=1, drop_last=True)
# batches = next(iter(train_loader))
# samples = batch.to_data_list()

# for batch in batches:
#     print(batch)
# print(type(train_loader))


# # print(obj[44798]) #this is a empty bbox index and taken as a demo for exception handling
# # print(obj[44798].bbox)




#checking the size of the data fragment
###############################################################################
# print(obj[12345])
print(obj[51279])

# print(type(obj[12345]))
# print(f"the size of the obtained data fragment is {get_data_size(obj[12345])} bytes")

# print(f"python's obj size checker {sys.getsizeof(obj[44780])}")
###############################################################################








