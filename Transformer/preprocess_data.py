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

root_path = "/mnt/home/data/giga2024/Trajectory/"
train_path = root_path + 'train/'
test_path = root_path + 'test/'
train_set = [train_path + 'annos/train_1.ndjson', train_path + 'annos/train_2.ndjson',
             train_path + 'annos/train_3.ndjson', train_path + 'annos/train_4.ndjson',
             train_path + 'annos/train_5.ndjson', train_path + 'annos/train_6.ndjson',
             train_path + 'annos/train_7.ndjson', train_path + 'annos/train_8.ndjson', 
            ]
test_set = [test_path + 'annos/test_1.ndjson', test_path + 'annos/test_2.ndjson',
            test_path + 'annos/test_3.ndjson', test_path + 'annos/test_4.ndjson', 
            test_path + 'annos/test_5.ndjson', test_path + 'annos/test_6.ndjson',
            test_path + 'annos/test_7.ndjson', test_path + 'annos/test_8.ndjson', 
           ]

train_files_path = train_path + 'transformer_preprocess_files/'
test_files_path = [test_path + 'transformer_preprocess_files/test_1/', 
                   test_path + 'transformer_preprocess_files/test_2/',
                   test_path + 'transformer_preprocess_files/test_3/', 
                   test_path + 'transformer_preprocess_files/test_4/',
                   test_path + 'transformer_preprocess_files/test_5/', 
                   test_path + 'transformer_preprocess_files/test_6/',
                   test_path + 'transformer_preprocess_files/test_7/', 
                   test_path + 'transformer_preprocess_files/test_8/',
             ]
vis_path = "/mnt/home/data/giga2024/Trajectory/vis/"
vector_repr = True  # a time step t is valid only when both t and t-1 are valid

def train():
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
                frame_track_pedestrian_id_lists = []
                observed_step = scene_data['end_frame'] - scene_data['start_frame'] + 1
                if observed_step < 62:
                    continue
                for s, frame in enumerate(range(scene_data['start_frame'], scene_data['end_frame']+1)):
                    frame_idx = track_frames == frame  
                    frame_track_pedestrian_id_list = track_pedestrian_ids[frame_idx]
                    frame_track_pedestrian_id_lists.append(frame_track_pedestrian_id_list)
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
                
                for agent_current_step in range(1, observed_step):
                    if agent_current_step + 60 >= observed_step and f_idx < 8:
                        continue
                    train_num = train_num + 1
                    agent_ids_list = []
                    for step in range(max(agent_current_step-59, 0), agent_current_step+1):
                        agent_ids_list = agent_ids_list + list(frame_track_pedestrian_id_lists[step])
                    agent_ids = list(set(agent_ids_list))
                    num_agents = len(agent_ids)
                    valid_mask = torch.zeros(num_agents, 120, dtype=torch.bool)
                    current_valid_mask = torch.zeros(num_agents, dtype=torch.bool)
                    predict_mask = torch.zeros(num_agents, 120, dtype=torch.bool)
                    agent_id: List[Optional[str]] = [None] * num_agents
                    position = torch.zeros(num_agents, 120, 2, dtype=torch.float)
                    for track_id in agent_ids:
                        agent_idx = agent_ids.index(track_id)
                        agent_steps = scene_pred_list[track_id]['step']
                        agent_steps = np.array(agent_steps) + (60 - agent_current_step - 1)
                        valid_step_mask = np.logical_and(agent_steps >= 0, agent_steps < 120)
                        valid_step = agent_steps[valid_step_mask]
                        position[agent_idx, valid_step] = scene_pred_list[track_id]['position'
                                                                ][valid_step_mask].float()
                        valid_mask[agent_idx, valid_step] = True
                        current_valid_mask[agent_idx] = valid_mask[agent_idx, 59]
                        predict_mask[agent_idx, valid_step] = True
                        if vector_repr:  # a time step t is valid only when both t and t-1 are valid
                            valid_mask[agent_idx, 1:60] = (
                            valid_mask[agent_idx, :59] &
                            valid_mask[agent_idx, 1:60])
                            valid_mask[agent_idx, 0] = False
                        predict_mask[agent_idx, :60] = False 
                        if not current_valid_mask[agent_idx]:
                            predict_mask[agent_idx, 60:] = False
                        agent_id[agent_idx] = track_id
                    data = dict()
                    data['num_nodes'] = num_agents
                    data['valid_mask'] = valid_mask
                    data['predict_mask'] = predict_mask
                    data['position'] = position
                    data['id'] = agent_id
                    train_file = train_files_path + str(f_idx) + "_"  + str(i) + "_" + str(agent_current_step
                                                                                          ) + ".pkl"
                    f = open(train_file, 'wb')
                    pickle.dump(copy.deepcopy(data), f, protocol=pickle.HIGHEST_PROTOCOL)
                    f.close()
    print(train_num)
    
def test():
    t = time.time()
    test_num = 0
    test_num_valid = 0
    for f_idx, file in enumerate(test_set):
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
                    test_num = test_num + 1
                    scene_ids.append(data[i]['scene']['id'])
                    scene_primary_pedestrian_ids.append(data[i]['scene']['p'])
                    scene_start_frames.append(data[i]['scene']['s'])
                    scene_end_frames.append(data[i]['scene']['e'])
                elif 'track' in data[i]:
                    num_track = num_track + 1
                    track_frames.append(data[i]['track']['f'])
                    track_pedestrian_ids.append(data[i]['track']['p'])
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
                for frame in range(scene_data['start_frame'], scene_data['end_frame']+1):
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
                        scene_pred_list[pred_id]['step'].append(frame-1)

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
                trajs = [agent_traj] + trajs
                steps = [agent_steps] + steps
                scene_pred_ids = [agent_scene_pred_ids] + scene_pred_ids
                if 59 not in steps[0]:
                    print("LLLLLLLLLLLLLLLLLLLLLLLLLLLcut")
                    continue      
                test_num_valid = test_num_valid + 1
                current_step_index = steps[0].tolist().index(59)
                pre_current_step_index = current_step_index-1
                orig = trajs[0][current_step_index].copy().astype(np.float32)
                pre = trajs[0][pre_current_step_index] - orig
                theta = np.pi - np.arctan2(pre[1], pre[0])
                rot = np.asarray([
                     [np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]], np.float32)
                
                feats, ctrs, gt_preds, has_preds, valid_track_ids = [], [], [], [], []
                origin_past_ctrs = []
                for traj, step, pred_id in zip(trajs, steps, scene_pred_ids):
                    if 59 not in step:
                        continue
                    valid_track_ids.append(pred_id)
                    gt_pred = np.zeros((60, 2), np.float32)
                    has_pred = np.zeros(60, bool)
                    future_mask = np.logical_and(step >= 60, step < 120)
                    future_step = step[future_mask] - 60
                    future_traj = traj[future_mask]
                    if vis_count < 10:
                        vis_future.append(future_traj)
                    gt_pred[future_step] = future_traj[:,:2]
                    has_pred[future_step] = 1
            
                    obs_mask = step < 60
                    step = step[obs_mask]
                    traj = traj[obs_mask]
                    if vis_count < 10:
                        vis_past.append(traj)
                    idcs = step.argsort()
                    step = step[idcs]
                    traj = traj[idcs]
                  
                    origin_past_ctr = np.zeros((60, 2), np.float32)
                    origin_past_ctr[step, :2] = traj[:, :2].copy()
                    origin_past_ctrs.append(origin_past_ctr)
                  
                    feat = np.zeros((60, 3), np.float32)
                    feat[step, :2] = np.matmul(rot, (traj[:, :2] - orig.reshape(-1, 2)).T).T
                    feat[step, 2] = 1.0
            
                    ctrs.append(feat[-1, :2].copy())
                    feat[1:, :2] -= feat[:-1, :2]
                    feat[step[0], :2] = 0
                    feats.append(feat)
                    gt_preds.append(gt_pred)
                    has_preds.append(has_pred)
                
                valid_track_ids = np.asarray(valid_track_ids, np.int64)
                feats = np.asarray(feats, np.float32)
                ctrs = np.asarray(ctrs, np.float32)
                origin_past_ctrs = np.asarray(origin_past_ctrs, np.float32)
                gt_preds = np.asarray(gt_preds, np.float32)
                has_preds = np.asarray(has_preds, bool)

                scene_data['track_ids'] = valid_track_ids
                scene_data['feats'] = feats
                scene_data['ctrs'] = ctrs
                scene_data['orig'] = orig
                scene_data['theta'] = theta
                scene_data['rot'] = rot
                scene_data['gt_preds'] = gt_preds
                scene_data['has_preds'] = has_preds
                scene_data['origin_past_ctrs'] = origin_past_ctrs
                test_file = test_files_path[f_idx] + str(f_idx) + "_"  + str(i) + ".pkl"
                f = open(test_file, 'wb')
                pickle.dump(copy.deepcopy(scene_data), f, protocol=pickle.HIGHEST_PROTOCOL)
                f.close()
                if vis_count < 10:
                    for vi in range(len(vis_future)):
                        color = 'blue'
                        lw = 1
                        if vi == 0:
                            color = 'green'
                            lw = 3
                        plt.plot(vis_future[vi][:,0], vis_future[vi][:,1], 
                                 "--", linewidth=lw, color=color)
                        plt.plot(vis_past[vi][:,0], vis_past[vi][:,1], 
                                 "-", linewidth=lw, color=color)
                    save_path = vis_path + "test_"+ str(f_idx) +  "_"  + str(i) + ".png"
                    plt.savefig(save_path)
                    plt.cla()
                vis_count = vis_count + 1

train()
#test()
