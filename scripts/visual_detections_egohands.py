import cv2
import argparse
import h5py
from pathlib import Path
import numpy as np
import os
from dsec_det.directory import DSECDirectory
from dsec_det.io import extract_from_h5_by_timewindow, extract_image_by_index, load_start_and_end_time
from dsec_det.preprocessing import compute_index

from dagr.visualization.bbox_viz import draw_bbox_on_img
from dagr.visualization.event_viz import draw_events_on_image




def load_time(dir):
    with h5py.File(str(f"{dir}/events/{args.sequence}.h5"), 'r') as fh:
        fh = fh["events"]
        t = fh["t"]
        return t[0], t[-1]


def get_frame(idx,frames_path):
        frame_path1 = os.path.join(frames_path,f"frame_{idx}.jpg")
        
        frame1 = cv2.imread(frame_path1)

        return frame1

def get_events(idx):
    #returns a dictionary of events in that timeframe
    
    # events_root = os.path.join("/cfs/earth/scratch/kotabha1/temp/events")
    rel_idx = idx
    # file_paths1 = []
    # for file in os.listdir(events_root):  # Iterate over all items in the 'frames' folder
    #     file_path = os.path.join(events_root, file)  # Get the full path of the item
    #     file_paths1.append(file_path)




    
    # # print(file_paths1)
    # file_paths = sorted(file_paths1)
    # # print(file_paths)
    file = "/cfs/earth/scratch/kotabha1/temp/events/jenga_courtyard_S_T.h5"
    with h5py.File(str(file)) as fh:
        events = fh["events"]
        timeidx = fh["timeidx"]
        x = events["x"]
        y = events["y"]
        t = events["t"]
        p = events["p"]
        timestamp1 = timeidx[3]
        if ((rel_idx+2)<len(timeidx)):
            timestamp2 = timeidx[4]
        else:
            

            return get_events(370)
        x_new = x[timestamp1:(timestamp2-1)].astype(np.uint16)
        p_new = p[timestamp1:(timestamp2-1)].astype(np.int64)
        # p_new = p_new.astype(np.int8)


        p_new = 2 * p_new.reshape((-1,1)) - 1
        # print("p_new after weird reshape",p_new)
        t_new = t[timestamp1:(timestamp2-1)].astype(np.int64)
        y_new = y[timestamp1:(timestamp2-1)].astype(np.uint16)




        
        
        events = {
            'p': p_new,
            't': t_new,
            'x': x_new,
            'y': y_new,
        }

    
    return events


if __name__ == '__main__':
    parser = argparse.ArgumentParser("""Visualization script to show bounding boxes""")
    parser.add_argument("--detections_folder", help="Path to folder with detections.", type=Path)
    parser.add_argument("--dataset_directory", help="Path to DSEC folder including which split.", type=Path, default="/data/scratch1/daniel/datasets/DSEC_fragment/test")
    parser.add_argument("--vis_time_step_us", help="Number of microseconds to step each iteration.", type=int, default=1000)
    parser.add_argument("--event_time_window_us", help="Length of sliding event time window for visualization.", type=int, default=5000)
    parser.add_argument("--sequence", help="Sequence to visualize. Must be an official DSEC sequence e.g. zurich_city_13_b", default="zurich_city_13_b", type=str)
    parser.add_argument("--write_to_output", help="Whether to save images in folder ${detections_folder}/visualization. Otherwise, just cv2.imshow is used.", action="store_true")
    args = parser.parse_args()

    assert args.dataset_directory.exists()
    assert args.vis_time_step_us > 0
    assert args.event_time_window_us > 0

    if args.write_to_output:
        assert (args.detections_folder / f"detections_{args.sequence}.npy").exists()
        assert args.detections_folder.exists()
        output_path = args.detections_folder / "visualization"
        output_path.mkdir(parents=True, exist_ok=True)

    dsec_directory = DSECDirectory(args.dataset_directory / args.sequence)

    # t0, t1 = load_start_and_end_time(dsec_directory)
    t0, t1 = load_time("/cfs/earth/scratch/kotabha1/temp")
    ego_timestamps = [int(1e6 * ((1 / 30) * i)) for i in range(2700)]
    vis_timestamps = np.arange(t0, t1, step=args.vis_time_step_us)
    step_index_to_image_index = compute_index(ego_timestamps, vis_timestamps)
    print("the step index to image index is", step_index_to_image_index)

    show_detections = args.detections_folder is not None

    if not show_detections:
        print("Did not specifiy detections. Just showing events and images.")

    if show_detections:
        detections_file = args.detections_folder / f"detections_{args.sequence}.npy"
        detections = np.load(detections_file)
        detection_timestamps = np.unique(detections['t'])
        step_index_to_boxes_index = compute_index(detection_timestamps, vis_timestamps)

    scale = 2

    for step, t in enumerate(vis_timestamps):

        # find most recent image
        image_index = step_index_to_image_index[step]
        image = get_frame(image_index,"/cfs/earth/scratch/kotabha1/temp/frames/jenga_courtyard_S_T")
        # print("the image is", image)

        # find events within time window [image_timestamps, t]
        events = get_events(image_index)
        # print("the events is", events['x'])
        print("the length of p is", len(events['p']))
        # print("the length of the image is", len(image))

        image = draw_events_on_image(image, events['x'], events['y'], events['p'])

        if show_detections:
            # find most recent bounding boxes
            boxes_index = step_index_to_boxes_index[step]
            boxes_timestamp = detection_timestamps[boxes_index]
            boxes = detections[detections['t'] == boxes_timestamp]

            # draw them on one image
            scale = 2
            image = draw_bbox_on_img(image, scale*boxes['x'], scale*boxes['y'], scale*boxes['w'], scale*boxes["h"],
                                     boxes["class_id"], boxes['class_confidence'], conf=0.3, nms=0.65)

        if args.write_to_output:
            cv2.imwrite(str(output_path / ("%06d.png" % step)), image)
        else:
            cv2.imshow("DSEC Det: Visualization", image)
            cv2.waitKey(3)

