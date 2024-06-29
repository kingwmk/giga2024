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

train_file = train_path + 'preprocess_light/train.p'
train_train_file = train_path + 'preprocess_light/train_train.p'
train_val_file = train_path + 'preprocess_light/train_val.p'
test_files = [test_path + 'preprocess/test_1.p', test_path + 'preprocess/test_2.p', 
              test_path + 'preprocess/test_3.p', test_path + 'preprocess/test_4.p', 
              test_path + 'preprocess/test_5.p', test_path + 'preprocess/test_6.p', 
              test_path + 'preprocess/test_7.p', test_path + 'preprocess/test_8.p', 
             ]

vis_path = "/mnt/home/data/giga2024/Trajectory/vis/"
def train(train_file, train_train_file, train_val_file):
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
            track_xcs = []
            track_ycs = []
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
                    track_xws.append(data[i]['track']['x_w'])
                    track_yws.append(data[i]['track']['y_w'])
                    track_xcs.append(data[i]['track']['x_c'])
                    track_ycs.append(data[i]['track']['y_c'])   
                  
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
                for s, frame in enumerate(range(scene_data['start_frame'],
                                                scene_data['end_frame']+1)):
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
                trajs = [agent_traj] + trajs
                steps = [agent_steps] + steps
                scene_pred_ids = [agent_scene_pred_ids] + scene_pred_ids
                for agent_current_step in range(1, agent_steps[-1]):
                    if agent_current_step + 60 > agent_steps[-1]:
                        continue
                    if (agent_current_step not in agent_steps) or ((agent_current_step-1) not in agent_steps):
                        continue
                    train_num = train_num + 1
                    current_step_index = agent_steps.tolist().index(agent_current_step)
                    pre_current_step_index = current_step_index-1
                    orig = agent_traj[current_step_index].copy().astype(np.float32)
                    pre = agent_traj[pre_current_step_index] - orig
                    theta = np.pi - np.arctan2(pre[1], pre[0])
                    rot = np.asarray([
                         [np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]], np.float32)
                
                    feats, ctrs, gt_preds, has_preds, valid_track_ids = [], [], [], [], []
                    for traj, step, pred_id in zip(trajs, steps, scene_pred_ids):
                        step = step + (60 - agent_current_step - 1)
                        if 59 not in step:
                            continue
                        valid_track_ids.append(pred_id)
                        gt_pred = np.zeros((60, 2), np.float32)
                        has_pred = np.zeros(60, bool)
                        future_mask = np.logical_and(step >= 60, 
                                                     step < 120)
                        future_step = step[future_mask] - 60
                        future_traj = traj[future_mask]
                        if vis_count < 10:
                            vis_future.append(future_traj)
                        gt_pred[future_step] = future_traj[:,:2]
                        has_pred[future_step] = 1
            
                        obs_mask = np.logical_and(step >= 0, step < 60)
                        step = step[obs_mask]
                        traj = traj[obs_mask]
                        if vis_count < 10:
                            vis_past.append(traj)
                        idcs = step.argsort()
                        step = step[idcs]
                        traj = traj[idcs]
            
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
                    gt_preds = np.asarray(gt_preds, np.float32)
                    has_preds = np.asarray(has_preds, bool)
                  
                    data = dict()
                    data['feats'] = feats
                    data['ctrs'] = ctrs
                    data['orig'] = orig
                    data['theta'] = theta
                    data['rot'] = rot
                    data['gt_preds'] = gt_preds[0:1]
                    data['has_preds'] = has_preds[0:1]
                    stores.append(copy.deepcopy(data))
                
                    if vis_count < 10:
                        for vi in range(len(vis_future)):
                            color = 'blue'
                            lw = 1
                            if vi == 0:
                                lw = 3
                                color = 'g'
                            plt.plot(vis_future[vi][:,0], vis_future[vi][:,1], 
                                 "--", linewidth=lw, color=color)
                            plt.plot(vis_past[vi][:,0], vis_past[vi][:,1], 
                                 "-", linewidth=lw, color=color)
                        save_path = vis_path + "train_"+ str(f_idx) + "_"  + str(i) + ".png"
                        plt.savefig(save_path)
                        plt.cla()
                    vis_count = vis_count + 1

    f = open(train_file, 'wb')
    pickle.dump(stores, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

    total_scene_num = len(stores)
    train_train_num = int(total_scene_num * train_val_ratio)
    train_val_num = total_scene_num - train_train_num
    stores_idx = range(0, total_scene_num)
    
    train_train_slice = random.sample(stores_idx, train_train_num)
    train_train_stores, train_val_stores = [], []
    for idx in range(total_scene_num):
        if idx in train_train_slice:
            train_train_stores.append(stores[idx])
        else:
            train_val_stores.append(stores[idx])

    f = open(train_train_file, 'wb')
    pickle.dump(train_train_stores, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
        
    f = open(train_val_file, 'wb')
    pickle.dump(train_val_stores, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

    print(train_num)
    
def test(test_files):
    t = time.time()
    test_num = 0
    test_num_valid = 0
    for f_idx,file in enumerate(test_set):
        vis_count = 0
        stores = []
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
                
                stores.append(scene_data)
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
        print(test_num)
        print(test_num_valid)       
        f = open(test_files[f_idx], 'wb')
        pickle.dump(stores, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

train(train_file, train_train_file, train_val_file)
#test(test_files)
