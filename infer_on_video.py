import traceback

from model import BallTrackerNet
import torch
import cv2
from general import postprocess, process_batch
from tqdm.auto import tqdm
import numpy as np
import argparse
from itertools import groupby
from scipy.spatial import distance
import time
from torch.utils.data import Dataset, DataLoader


# set number of intra op threads == 4
# torch.set_num_threads(2)
# torch.set_num_interop_threads(2)

# Dataloader class
class trackNetDataset(Dataset):
    def __init__(self, image_list, width=640, height=360):
        self.image_list = image_list
        # self.return_image_list = image_list[start_index: ]
        self.width = width
        self.height = height

    def __len__(self):
        return len(self.image_list)

    def process_images(self, idx):
        img = cv2.resize(self.image_list[idx], (self.width, self.height))
        img_prev = cv2.resize(self.image_list[idx - 1], (self.width, self.height))
        img_preprev = cv2.resize(self.image_list[idx - 2], (self.width, self.height))
        if idx < 2:
            imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        else:
            imgs = np.concatenate((img, img, img), axis=2)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0)
        # inp = np.expand_dims(imgs, axis=0)
        return imgs

    def __getitem__(self, idx):
        processed_img = self.process_images(idx)
        return processed_img


def read_video(path_video):
    """ Read video file    
    :params
        path_video: path to video file
    :return
        frames: list of video frames
        fps: frames per second
    """
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, fps

def infer_model(frames, model, log_file, args):
    """ Run pretrained model on a consecutive list of frames    
    :params
        frames: list of consecutive video frames
        model: pretrained model
    :return    
        ball_track: list of detected ball points
        dists: list of euclidean distances between two neighbouring ball points
    """
    height = 360
    width = 640
    dists = [-1]*2
    ball_track = [(None,None)]*2
    out_frames = []
    model_results = open('../model_results.txt', 'w+')
    infer_dataset = trackNetDataset(frames)
    infer_dataloader = DataLoader(infer_dataset, batch_size=args.batch_size)

    # for num in tqdm(range(2, len(frames))):
        # img = cv2.resize(frames[num], (width, height))
        # img_prev = cv2.resize(frames[num-1], (width, height))
        # img_preprev = cv2.resize(frames[num-2], (width, height))
        # imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        # imgs = imgs.astype(np.float32)/255.0
        # imgs = np.rollaxis(imgs, 2, 0)
        # inp = np.expand_dims(imgs, axis=0)
    start_idx = 0
    for i, batch in tqdm(enumerate(infer_dataloader)):
        inf_start = time.time()
        # out = model(torch.from_numpy(batch).float().to(device))
        with torch.no_grad():
            out = model(batch.to(device))
        inf_end = time.time()
        model_results.write(f'Batch: {i} \n {out} \n {out.shape} \n \n')
        # output = out.argmax(dim=1).detach().cpu().numpy()
        # try:
        #     x_pred, y_pred = postprocess(output)
        #     # post_process = time.time()
        #     # log_file.write(f'Model inference iter {i} inference time: {inf_end - inf_start} post process time: {post_process - inf_end} \n')
        #     ball_track.append((x_pred, y_pred))
        # except:
        #     print(traceback.format_exc())
        #     continue
        # for j in range(start_idx, start_idx + args.batch_size):
        #     img = cv2.resize(frames[j], (width, height))
        #     out_frames.append(img)
        # if ball_track[-1][0] and ball_track[-2][0]:
        #     dist = distance.euclidean(ball_track[-1], ball_track[-2])
        # else:
        #     dist = -1
        # dists.append(dist)

        output = out.detach().cpu().numpy()
        try:
            batch_results = process_batch(output)
            for x_pred, y_pred in batch_results:
                ball_track.append((x_pred, y_pred))
                if ball_track[-1][0] is not None and ball_track[-2][0] is not None:
                    dist = distance.euclidean(ball_track[-1], ball_track[-2])
                else:
                    dist = -1
                dists.append(dist)
        except Exception as e:
            print(f"Error in postprocessing for batch: {i}: {str(e)}")
            print(traceback.format_exc())
            continue

        for j in range(start_idx, min(start_idx + args.batch_size, len(frames))):
            img = cv2.resize(frames[j], (width, height))
            out_frames.append(img)

        start_idx += args.batch_size

        post_process = time.time()
        log_file.write(
            f'Batch {i} inference time: {inf_end - inf_start}, post process time: {post_process - inf_end}\n')

    return ball_track, dists, out_frames

def remove_outliers(ball_track, dists, max_dist = 20):
    """ Remove outliers from model prediction    
    :params
        ball_track: list of detected ball points
        dists: list of euclidean distances between two neighbouring ball points
        max_dist: maximum distance between two neighbouring ball points
    :return
        ball_track: list of ball points
    """
    outliers = list(np.where(np.array(dists) > max_dist)[0])
    for i in outliers:
        if (dists[i+1] > max_dist) | (dists[i+1] == -1):       
            ball_track[i] = (None, None)
            outliers.remove(i)
        elif dists[i-1] == -1:
            ball_track[i-1] = (None, None)
    return ball_track  

def split_track(ball_track, max_gap=4, max_dist_gap=30, min_track=5):
    """ Split ball track into several subtracks in each of which we will perform
    ball interpolation.    
    :params
        ball_track: list of detected ball points
        max_gap: maximun number of coherent None values for interpolation  
        max_dist_gap: maximum distance at which neighboring points remain in one subtrack
        min_track: minimum number of frames in each subtrack    
    :return
        result: list of subtrack indexes    
    """
    list_det = [0 if x[0] else 1 for x in ball_track]
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

    cursor = 0
    min_value = 0
    result = []
    for i, (k, l) in enumerate(groups):
        if (k == 1) & (i > 0) & (i < len(groups) - 1):
            dist = distance.euclidean(ball_track[cursor-1], ball_track[cursor+l])
            if (l >=max_gap) | (dist/l > max_dist_gap):
                if cursor - min_value > min_track:
                    result.append([min_value, cursor])
                    min_value = cursor + l - 1        
        cursor += l
    if len(list_det) - min_value > min_track: 
        result.append([min_value, len(list_det)]) 
    return result    

def interpolation(coords):
    """ Run ball interpolation in one subtrack    
    :params
        coords: list of ball coordinates of one subtrack    
    :return
        track: list of interpolated ball coordinates of one subtrack
    """
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    x = np.array([x[0] if x[0] is not None else np.nan for x in coords])
    y = np.array([x[1] if x[1] is not None else np.nan for x in coords])

    nons, yy = nan_helper(x)
    x[nons]= np.interp(yy(nons), yy(~nons), x[~nons])
    nans, xx = nan_helper(y)
    y[nans]= np.interp(xx(nans), xx(~nans), y[~nans])

    track = [*zip(x,y)]
    return track

def write_track(frames, ball_track, path_output_video, fps, trace=7):
    """ Write .avi file with detected ball tracks
    :params
        frames: list of original video frames
        ball_track: list of ball coordinates
        path_output_video: path to output video
        fps: frames per second
        trace: number of frames with detected trace
    """
    height, width = frames[0].shape[:2]
    # out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'DIVX'),
    #                       fps, (width, height))
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                          fps, (width, height))
    for num in range(len(frames)):
        frame = frames[num]
        for i in range(trace):
            if (num-i > 0):
                if ball_track[num-i][0]:
                    x = int(ball_track[num-i][0])
                    y = int(ball_track[num-i][1])
                    frame = cv2.circle(frame, (x,y), radius=0, color=(0, 0, 255), thickness=10-i)
                else:
                    break
        out.write(frame) 
    out.release()    

if __name__ == '__main__':
    log_file = open('../model_inference_times.txt', 'w+')
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--video_path', type=str, help='path to input video')
    parser.add_argument('--video_out_path', type=str, help='path to output video')
    parser.add_argument('--extrapolation', action='store_true', help='whether to use ball track extrapolation')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    arg_time = time.time()
    log_file.write(f'time taken for arg parsing {arg_time - start_time} \n \n')
    
    model = BallTrackerNet()
    device = args.device
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    load_time = time.time()
    log_file.write(f'time taken to load model {load_time - arg_time} \n \n')
    
    frames, fps = read_video(args.video_path)
    read_video_time = time.time()
    log_file.write(f'time taken to read video {read_video_time - load_time} \n \n')
    ball_track, dists, out_frames = infer_model(frames, model, log_file, args)
    inference_time = time.time()
    log_file.write(f'{inference_time - read_video_time} \n \n')
    with open('ball_track_raw.txt', 'w+') as f:
        f.write(str(ball_track))
        f.close()
    save_time = time.time()
    log_file.write(f'time taken to save ball track file: {save_time - inference_time} \n \n')
    ball_track = remove_outliers(ball_track, dists)
    remove_out_time = time.time()
    log_file.write(f'time taken to remove outliers {remove_out_time - save_time} \n \n')
    with open('ball_track_outlier.txt', 'w+') as f:
        f.write(str(ball_track))
        f.close()
    save_outlier = time.time()
    log_file.write(f'time taken to save outliers {save_outlier - remove_out_time}')
    with open('ball_dist.txt', 'w+') as f:
        f.write(str(dists))
        f.close()
    save_dist = time.time()
    log_file.write(f'time to taken save dist {save_dist - save_outlier} \n \n')
    if args.extrapolation:
        subtracks = split_track(ball_track)
        split_track_time = time.time()
        log_file.write(f'split track time {split_track_time - save_dist} \n \n')
        with open('subtracks.txt', 'w+') as f:
            f.write(str(subtracks))
            f.close()
        for r in subtracks:
            ball_subtrack = ball_track[r[0]:r[1]]
            ball_subtrack = interpolation(ball_subtrack)
            ball_track[r[0]:r[1]] = ball_subtrack
        interp_time = time.time()
        log_file.write(f'interpolation time {interp_time - split_track_time} \n \n')
        with open('ball_track_extrapolated.txt', 'w+') as f:
            f.write(str(ball_track))
            f.close()

    log_file.write(f'Total time taken: {interp_time - start_time}')
    write_track(frames, ball_track, args.video_out_path, fps)
    log_file.close()
    
    
    
    
    
