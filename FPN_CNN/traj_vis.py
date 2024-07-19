import ndjson
import os
import numpy as np
import sys
import time
import pickle
from skimage.transform import rotate
import random
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
import random  

root_path = "/mnt/home/data/giga2024/Trajectory/"
train_path = root_path + 'train/'
test_path = root_path + 'test/'
train_set = [train_path + 'annos/train_1.ndjson', train_path + 'annos/train_2.ndjson',
             train_path + 'annos/train_3.ndjson', train_path + 'annos/train_4.ndjson',
             train_path + 'annos/train_5.ndjson', train_path + 'annos/train_6.ndjson',
             train_path + 'annos/train_7.ndjson', train_path + 'annos/train_8.ndjson', 
             test_path + 'annos/test_1.ndjson', test_path + 'annos/test_2.ndjson',
             test_path + 'annos/test_3.ndjson', test_path + 'annos/test_4.ndjson', 
             test_path + 'annos/test_5.ndjson', test_path + 'annos/test_6.ndjson',
             test_path + 'annos/test_7.ndjson', test_path + 'annos/test_8.ndjson',
            ]

vis_path = "/mnt/home/data/giga2024/Trajectory/traj_vis/"
def train():
    stores = []
    train_val_ratio = 0.8
    t = time.time()
    train_num = 0
    for f_idx, file in enumerate(train_set):
        vis_count = 0
        with open(file) as f:
            data = ndjson.load(f)
            num_scene = 0
            scene_ids = []
            scene_primary_pedestrian_ids = []
            scene_start_frames = []
            scene_end_frames = []
            
            num_track = 0
            track_frames = []
            track_pedestrian_ids = []
            track_xws = []
            track_yws = []
            for i in range(len(data)):
                if 'scene' in data[i]:
                    num_scene = num_scene + 1
                    scene_ids.append(data[i]['scene']['id'])
                    scene_primary_pedestrian_ids.append(data[i]['scene']['p'])
                    scene_start_frames.append(data[i]['scene']['s'])
                    scene_end_frames.append(data[i]['scene']['e'])
                elif 'track' in data[i]:
                    num_track = num_track + 1
                    track_frames.append(data[i]['track']['f'])
                    track_pedestrian_ids.append(data[i]['track']['p'])
                    if f_idx < 8:
                        track_xws.append(data[i]['track']['x_w'])
                        track_yws.append(data[i]['track']['y_w'])
                    else:
                        track_xws.append(data[i]['track']['x'])
                        track_yws.append(data[i]['track']['y'])  
                  
            track_frames = np.asarray(track_frames, np.int64)
            track_pedestrian_ids = np.asarray(track_pedestrian_ids, np.int64)
            track_xs = np.asarray(track_xws, np.float32)
            track_ys = np.asarray(track_yws, np.float32)
            
            for i in tqdm(range(num_scene)):
                vis_future, vis_past = [], []
                scene_data = dict()
                scene_data['scene_id'] = scene_ids[i]
                scene_data['scene_primary_pedestrian_id'] = scene_primary_pedestrian_ids[i]
                scene_data['start_frame'] = scene_start_frames[i]
                scene_data['end_frame'] = scene_end_frames[i]
                scene_pred_list = dict()
                for s, frame in enumerate(range(scene_data['start_frame'], scene_data['end_frame']+1)):
                    frame_idx = track_frames == frame  
                    frame_track_pedestrian_id_list = track_pedestrian_ids[frame_idx]
                    frame_track_x = track_xs[frame_idx]
                    frame_track_y = track_ys[frame_idx]
                    for j, pred_id in enumerate(frame_track_pedestrian_id_list):
                        if pred_id not in scene_pred_list:
                            pred = dict()
                            pred['position'] = []
                            pred['step'] = []
                            scene_pred_list[pred_id] = pred
                        scene_pred_list[pred_id]['position'].append([frame_track_x[j],
                                                                  frame_track_y[j]])
                        scene_pred_list[pred_id]['step'].append(s)

                trajs, steps, scene_pred_ids = [], [], []
                for pred_key in scene_pred_list.keys():
                    if scene_data['scene_primary_pedestrian_id'] == pred_key:
                        agent_traj = np.array(scene_pred_list[pred_key]['position'], np.float32)
                        agent_steps = np.array(scene_pred_list[pred_key]['step'], np.int64)
                        agent_scene_pred_ids = pred_key
                    else:
                        actor_traj = np.array(scene_pred_list[pred_key]['position'], np.float32)
                        actor_steps = np.array(scene_pred_list[pred_key]['step'], np.int64)
                        trajs.append(actor_traj)
                        steps.append(actor_steps)
                        scene_pred_ids.append(pred_key) 
                      
                plt.scatter(agent_traj[0,0], agent_traj[0,1], linewidth=1.6, color='red')
                plt.plot(agent_traj[:,0], agent_traj[:,1], "-", linewidth=1, color='green')
                plt.scatter(agent_traj[-1,0], agent_traj[-1,1], linewidth=1.6, color='black')
                save_path = vis_path + str(f_idx) + "_agent_"  + str(agent_scene_pred_ids) + "_" + str(len(agent_traj)) + ".png"
                plt.savefig(save_path)
                plt.cla()

train()

