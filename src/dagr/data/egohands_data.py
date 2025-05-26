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
from tqdm import tqdm
from PIL import Image
import glob

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
        self.height = 360
        self.width = 640
        self.num_classes = 1
        self.scale = 2
        self.num_us = -1
        #assert frames in dataset to ensure completeness
        self.assert_dataset_egohands()
        

        
        with open(self.bbox_path, "r") as f:
            self.tracks_data = json.load(f)

 
    def set_num_us(self, num_us):
        self.num_us = num_us


    def __len__(self):
        len = (self.num_frames-1)*(self.get_num_files(self.root,"events"))
        return len


    def __getitem__(self, idx):


        
        # Retrieve the frames, events, and tracks
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
        # print("tracks0 is",tracks0)




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
        # print("tracks0 is",tracks0)
        # print("tracks1 is",tracks1)

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
        # print("the data is",data)
        return data




    def assert_dataset_egohands(self):
        data_dir = os.path.dirname(self.root)
        folders = ["test", "train", "valid"]

        # Get the list of video names without extensions
        raw_videos_path = os.path.join(data_dir, "videos")
        videos_paths = glob.glob(f"{raw_videos_path}/*.mp4")
        videos = []
        for vid in videos_paths:
            vid_name = os.path.basename(vid)  # Get the base name of the video
            vid_name_without_extension = os.path.splitext(vid_name)[0]  # Remove the file extension
            videos.append(vid_name_without_extension)
        

        # # Iterate over the folders to check frame names
        # for folder in folders:
        #     root_dir = os.path.join(data_dir, folder, "frames")

        #     # Get all frame paths in the directory
        #     frame_paths = glob.glob(f"{root_dir}/*")
            
        #     # Set to track seen frame names and avoid duplicates
        #     seen_frame_names = set()

        #     for frame_path in frame_paths:
        #         frame_name = os.path.basename(frame_path)
        #         frame_name_without_extension = os.path.splitext(frame_name)[0]

        #         # Ensure no duplicates in the folder
        #         assert frame_name_without_extension not in seen_frame_names, \
        #             f"Error: Duplicate frame name '{frame_name_without_extension}' found in folder '{folder}'."
        #         seen_frame_names.add(frame_name_without_extension)

        #         # Assert that the frame name is in the list of video names
        #         assert frame_name_without_extension in videos, \
        #             f"Error: {frame_name_without_extension} is not in the list of videos."

        #     print(f"✅ All frame names in {folder} are valid video names and there are no duplicates.")


        #     # Iterate through each video folder with tqdm for a progress bar
        #     for vid_folder in tqdm(os.listdir(root_dir), desc=f"Checking {folder} folders", unit="video"):
        #         vid_path = os.path.join(root_dir, vid_folder)

        #         # Ensure it's a directory
        #         if os.path.isdir(vid_path):
        #             num_images = len([f for f in os.listdir(vid_path) if os.path.isfile(os.path.join(vid_path, f))])

        #             # Assert condition
        #             assert num_images == 2700, f"Error: {vid_folder} has {num_images} images instead of 2700"

        #             # Check frame names and resolutions
        #             for i in range(2700):
        #                 expected_filename = f"frame_{i}.jpg"
        #                 file_path = os.path.join(vid_path, expected_filename)
                        
        #                 # Ensure the file exists and is named correctly
        #                 assert os.path.isfile(file_path), f"Error: {file_path} does not exist"

        #                 # Check if the file name matches the expected frame name
        #                 actual_filename = os.path.basename(file_path)
        #                 assert actual_filename == f"frame_{i}.jpg", f"Error: {file_path} is incorrectly named {actual_filename}"

        #                 # Check the resolution of each image
        #                 img = Image.open(file_path)
        #                 width, height = img.size
        #                 assert width == 1280/self.scale and height == 720/self.scale, f"Error: {file_path} has incorrect resolution {width}x{height}"

        #     print(f"✅ All video folders in {folder} have exactly 2700 images, correctly named, and with the correct resolution.")






        for folder in folders:
            root_dir = os.path.join(data_dir, folder, "events")

            for file_path in tqdm(glob.glob(f"{root_dir}/*.h5")):
                with h5py.File(file_path, 'r') as f:
                    # Check the primary keys
                    keys_primary = list(f.keys())
                    required_primary_keys = ['events', 'timeidx']
                    
                    # Assert required primary keys are present
                    for key in required_primary_keys:
                        assert key in keys_primary, f"File '{file_path}': Key '{key}' not found in the primary keys."
                    
                    # Assert no extra primary keys are present
                    extra_primary_keys = set(keys_primary) - set(required_primary_keys)
                    assert not extra_primary_keys, f"File '{file_path}': Extra primary keys found: {extra_primary_keys}"
                    
                    # Access the "events" group and check secondary keys
                    if 'events' in keys_primary:
                        fh = f["events"]
                        keys_secondary = list(fh.keys())
                        required_secondary_keys = ['x', 'y', 't', 'p']
                        
                        # Assert required secondary keys are present
                        for key in required_secondary_keys:
                            assert key in keys_secondary, f"File '{file_path}': Key '{key}' not found in the secondary keys."
                        
                        # Assert no extra secondary keys are present
                        extra_secondary_keys = set(keys_secondary) - set(required_secondary_keys)
                        assert not extra_secondary_keys, f"File '{file_path}': Extra secondary keys found: {extra_secondary_keys}"

                    else:
                        print(f"File '{file_path}': 'events' group not found.")

        print(f"✅ The event files look good!")





    





    def get_events(self, idx):
        #returns a dictionary of events in that timeframe
        vid_no = self.get_vid_no(idx)
        events_root = os.path.join(self.root,"events")
        rel_idx = self.get_rel_idx(idx)
        file_paths1 = []
        for file in os.listdir(events_root):  # Iterate over all items in the 'frames' folder
            file_path = os.path.join(events_root, file)  # Get the full path of the item
            file_paths1.append(file_path)


        if rel_idx>2500:
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
                
                return self.get_events(370)

            x_new = x[timestamp1:(timestamp2-1)].astype(np.uint16)
            p_new = p[timestamp1:(timestamp2-1)]
            p_new = p_new.astype(np.int8)


            p_new = 2 * p_new.reshape((-1,1)) - 1
     
            t_new = t[timestamp1:(timestamp2-1)].astype(np.int64)
            y_new = y[timestamp1:(timestamp2-1)].astype(np.uint16)




            #outlier handling case
            if(len(x_new)<300):

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

    def rescale_tracks(self,tracks):
        
        rescaled_tracks=[]
        for track in tracks:
            # print("the track is",track)
            rescaled_track = []
            for i in track:
                i = i/self.scale
                rescaled_track.append(i)
                # print("the rescaled track is",rescaled_track)
            rescaled_tracks.append(rescaled_track)
        # print("the rescaled tracks are",rescaled_tracks)
        

        return rescaled_tracks



    def get_tracks_COCO(self, idx, bbox_path, vid_name: str,bbox_data):
        frame_idx = self.get_rel_idx(idx)
        # print("the index is",idx)
        # print("the relative index is", frame_idx)
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
        # print("the type of the bounding box is",type(bboxes0))
        # print("Before rescaling0", bboxes0)
        # print("Before rescaling1", bboxes1)
        bboxes0 = self.rescale_tracks(bboxes0)
        bboxes1 = self.rescale_tracks(bboxes1)
        # print("after rescaling0",bboxes0)
        # print("after rescaling1",bboxes1)
        # print("the bounding box0 is",bboxes0)
        # print("the bounding box1 is",bboxes1)
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

        # print("after yolo normalisation0",bbox0)
        # print("after yolo normalisation1",bbox1)

        # print("before normal full",bboxes0)
        # print("before normal full",bboxes1)

        # print("after normal", bbox0)
        # print("after bnormal",bbox1)

        #fine till here
        bbox0_new = []
        bbox1_new = []

        default = [[137, 363, 261, 352 ]]
        default = self.rescale_tracks(default)
        default[0].append(1)
        default = default[0]
        for i in bbox0:
            try:
                # print("the i0 being checked is",i)
                x = pbx.convert_bbox(i,from_type="yolo",to_type="coco",image_size=(self.width,self.height),strict=True)
                lst = list(x)  # Convert tuple to list
                lst.append(1)
                # print("passed")
                bbox0_new.append(lst)
                # bbox0_new = np.hstack([bbox0_new, np.ones((bbox0_new.shape[0], 1),dtype=np.float32)])


            except Exception as e:
                # print(f"the error converting the bounding box0 occured here  {e} and the index is {idx}")
                bbox0_new.append(default)



        for i in bbox1:
            try:
                # print("the i1 being checked is",i)
                x = pbx.convert_bbox(i,from_type="yolo",to_type="coco",image_size=(self.width,self.height),strict=True)
                lst = list(x)  # Convert tuple to list
                lst.append(1)
                # print("passed")
                bbox1_new.append(lst)
                # bbox1_new = np.hstack([bbox1_new, np.ones((bbox1_new.shape[0], 1),dtype=np.float32)])

            except Exception as e:
                # print(f"the error converting the bounding box1 occured here {e} and the index is {idx}")
                bbox1_new.append(default)


        # print("bbox0 before the error",bbox0_new)
        bbox0_new = np.array(bbox0_new)
        bbox1_new = np.array(bbox1_new)


        if bbox0_new.size == 0:
            bbox0_new = np.array[[137., 363., 261., 352., 1]]
            bbox0_new = self.rescale_tracks(bbox0_new)
            bbox0_new = np.array(bbox0_new)

        if bbox1_new.size == 0:
            bbox1_new = np.array[[137., 363., 261., 352., 1]]
            bbox1_new = self.rescale_tracks(bbox1_new)
            bbox1_new = np.array(bbox1_new)

        # print("before stacking0",bbox0_new)
        # print("before stacking1",bbox1_new)

        # bbox0_new = np.hstack([bbox0_new, np.ones((bbox0_new.shape[0], 1),dtype=np.float32)])
        # bbox1_new = np.hstack([bbox1_new, np.ones((bbox1_new.shape[0], 1),dtype=np.float32)])
        return bbox0_new, bbox1_new
    




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
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = image.unsqueeze(0)
        
        return image




#calls and tests
# path = "/cfs/earth/scratch/kotabha1/egohands/test/events/chess_livingroom_S_B.h5"
root = "/cfs/earth/scratch/kotabha1/egohands_v3/train"
bbox_path = "/cfs/earth/scratch/kotabha1/egohands_v3/bounding_boxes.json"
root_test = "/cfs/earth/scratch/kotabha1/egohands_v3/test"
root_train = "/cfs/earth/scratch/kotabha1/egohands_v3/train"








###############################################################################
# train_dataset = Egohands(root_train, bbox_path)
obj = Egohands(root,bbox_path)
print(obj[419])

###############################################################################








