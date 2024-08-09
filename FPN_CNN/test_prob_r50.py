import argparse
import os
os.umask(0)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import pickle
import sys
from importlib import import_module
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import numpy as np
from utils import Logger, load_pretrain
from typing import Dict, Final, List, Tuple
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)
import ndjson
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
from scipy.special import softmax
from scipy.ndimage import gaussian_filter1d

# define parser
parser = argparse.ArgumentParser(description="giga")
parser.add_argument(
    "-m1", "--model1", default="gigaNet_stride_6_test_rand_dcms3_train_lr_430", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight1", default="3.310.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m2", "--model2", default="gigaNet_stride_6_test_rand_dcms3_train_r50", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight2", default="3.210.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m3", "--model3", default="gigaNet_stride_6_test_rand_dcms3_train_r50", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight3", default="1.003.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m4", "--model4", default="gigaNet_stride_6_test_rand_dcms_train", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight4", default="2.508.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m5", "--model5", default="gigaNet_stride_6_test_rand_dcms3_train_lr", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight5", default="4.514.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m6", "--model6", default="gigaNet_stride_6_test_rand_dcms_train", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight6", default="2.006.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)

root_path = "/mnt/home/data/giga2024/Trajectory/"
test_data_path = root_path + "test/"

test_files = [test_data_path + 'preprocess_stride_6_test_dcms/test_1.p',
              test_data_path + 'preprocess_stride_6_test_dcms/test_2.p',
              test_data_path + 'preprocess_stride_6_test_dcms/test_3.p', 
              test_data_path + 'preprocess_stride_6_test_dcms/test_4.p',
              test_data_path + 'preprocess_stride_6_test_dcms/test_5.p', 
              test_data_path + 'preprocess_stride_6_test_dcms/test_6.p', 
              test_data_path + 'preprocess_stride_6_test_dcms/test_7.p',
              test_data_path + 'preprocess_stride_6_test_dcms/test_8.p', 
             ]
    
output_files = ['submission/results/results/test_1.ndjson',
                'submission/results/results/test_2.ndjson',
                'submission/results/results/test_3.ndjson', 
                'submission/results/results/test_4.ndjson',
                'submission/results/results/test_5.ndjson', 
                'submission/results/results/test_6.ndjson',
                'submission/results/results/test_7.ndjson', 
                'submission/results/results/test_8.ndjson',
               ]

import random
def random_color():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color

def main():
    # Import all settings for experiment.
    args = parser.parse_args()

    #model1
    model1 = import_module(args.model1)
    config1, gigaDataset, collate_fn1, net1, loss1, post_process1, opt1 = model1.get_model()

    # load pretrain model
    ckpt_path1 = args.weight1
    if not os.path.isabs(ckpt_path1):
        ckpt_path1 = os.path.join(config1["save_dir"], ckpt_path1)
    ckpt1 = torch.load(ckpt_path1, map_location=lambda storage, loc: storage)
    load_pretrain(net1, ckpt1["state_dict"])
    net1.eval()
    
    #model2
    model2 = import_module(args.model2)
    config2, _, collate_fn2, net2, loss2, post_process2, opt2 = model2.get_model()

    # load pretrain model
    ckpt_path2 = args.weight2
    if not os.path.isabs(ckpt_path2):
        ckpt_path2 = os.path.join(config2["save_dir"], ckpt_path2)
    ckpt2 = torch.load(ckpt_path2, map_location=lambda storage, loc: storage)
    load_pretrain(net2, ckpt2["state_dict"])
    net2.eval()
    
    #model3
    model3 = import_module(args.model3)
    config3, _, collate_fn3, net3, loss3, post_process3, opt3 = model3.get_model()

    # load pretrain model
    ckpt_path3 = args.weight3
    if not os.path.isabs(ckpt_path3):
        ckpt_path3 = os.path.join(config3["save_dir"], ckpt_path3)
    ckpt3 = torch.load(ckpt_path3, map_location=lambda storage, loc: storage)
    load_pretrain(net3, ckpt3["state_dict"])
    net3.eval()
    
    #model4
    model4 = import_module(args.model4)
    config4, _, collate_fn4, net4, loss4, post_process4, opt4 = model4.get_model()

    # load pretrain model
    ckpt_path4 = args.weight4
    if not os.path.isabs(ckpt_path4):
        ckpt_path4 = os.path.join(config4["save_dir"], ckpt_path4)
    ckpt4 = torch.load(ckpt_path4, map_location=lambda storage, loc: storage)
    load_pretrain(net4, ckpt4["state_dict"])
    net4.eval()
    
    #model5
    model5 = import_module(args.model5)
    config5, _, collate_fn5, net5, loss5, post_process5, opt5 = model5.get_model()

    # load pretrain model
    ckpt_path5 = args.weight5
    if not os.path.isabs(ckpt_path5):
        ckpt_path5 = os.path.join(config5["save_dir"], ckpt_path5)
    ckpt5 = torch.load(ckpt_path5, map_location=lambda storage, loc: storage)
    load_pretrain(net5, ckpt5["state_dict"])
    net5.eval()
    
    #model6
    model6 = import_module(args.model6)
    config6, _, collate_fn6, net6, loss6, post_process6, opt6 = model6.get_model()

    # load pretrain model
    ckpt_path6 = args.weight6
    if not os.path.isabs(ckpt_path6):
        ckpt_path6 = os.path.join(config6["save_dir"], ckpt_path6)
    ckpt6 = torch.load(ckpt_path6, map_location=lambda storage, loc: storage)
    load_pretrain(net6, ckpt6["state_dict"])
    net6.eval()

    for f_idx in range(len(test_files)):
        vis_results, vis_assemble_results, vis_result_centers, vis_gt_pasts, vis_pp_ids = [], [], [], [], []
        #10 vis sample per test_file
        vis_count = 0
        file = open(output_files[f_idx], 'w')
        writer = ndjson.writer(file, ensure_ascii=False)
        # Data loader for evaluation
        dataset = gigaDataset(test_files[f_idx])
        data_loader = DataLoader(
            dataset,
            batch_size=config1["val_batch_size"],
            num_workers=config1["val_workers"],
            collate_fn=collate_fn1,
            shuffle=False,
            pin_memory=True,
        )

        # begin inference
        for ii, data in tqdm(enumerate(data_loader)):
            data = dict(data)
            with torch.no_grad():
                scene_ids = [x for x in data["scene_id"]]
                scene_primary_pedestrian_ids = [x for x in data["scene_primary_pedestrian_id"]]
                start_frames = [x for x in data["start_frame"]]
                end_frames = [x for x in data["end_frame"]]
                track_ids = [x[0].numpy().astype(np.int64) for x in data["track_ids"]]
                assert len(scene_primary_pedestrian_ids) == len(track_ids)
                for pred_id in range(len(scene_primary_pedestrian_ids)):
                    assert scene_primary_pedestrian_ids[pred_id] == track_ids[pred_id]

                output1 = net1(data)
                results1 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output1["reg"]]
                cls1 = [x.detach().cpu().numpy().astype(np.float64) for x in output1["cls"]]
                cls1 = softmax(cls1, axis=-1)
                output2 = net2(data)
                results2 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output2["reg"]]
                cls2 = [x.detach().cpu().numpy().astype(np.float64) for x in output2["cls"]]
                cls2 = softmax(cls2, axis=-1)
                output3 = net3(data)
                results3 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output3["reg"]]
                cls3 = [x.detach().cpu().numpy().astype(np.float64) for x in output3["cls"]]
                cls3 = softmax(cls3, axis=-1)
                output4 = net4(data)
                results4 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output4["reg"]]
                cls4 = [x.detach().cpu().numpy().astype(np.float64) for x in output4["cls"]]
                cls4 = softmax(cls4, axis=-1)
                output5 = net5(data)
                results5 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output5["reg"]]
                cls5 = [x.detach().cpu().numpy().astype(np.float64) for x in output5["cls"]]
                cls5 = softmax(cls5, axis=-1)
                output6 = net6(data)
                results6 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output6["reg"]]
                cls6 = [x.detach().cpu().numpy().astype(np.float64) for x in output6["cls"]]
                cls6 = softmax(cls6, axis=-1)
                
                results = []
                gt_pasts = [x[0].cpu().numpy().astype(np.float64) for x in data["origin_past_ctrs"]]
                for i in range(len(results1)):
                    trajs = np.concatenate((results1[i],results2[i],results3[i],results4[i],
                                            results5[i],results6[i],
                                           ), 1).squeeze()
                    probs = np.concatenate((cls1[i],cls2[i],cls3[i],cls4[i],
                                            cls5[i],cls6[i],
                                           ), 0).squeeze()
                    preds = trajs.squeeze()
                    sigma = 20
                    preds = np.array([gaussian_smoothing(preds[s], sigma) for s in range(preds.shape[0])])
                    window_size = 10
                    gt_past_cat = np.repeat(np.expand_dims(gt_pasts[i][58:59,:2], axis=0), repeats=preds.shape[0], 
                                            axis=0)
                    pred_gt_cat = np.concatenate([gt_past_cat, preds], axis=1)
                    pred_gt_cat = np.array([moving_average(pred_gt_cat[s], window_size) 
                                  for s in range(pred_gt_cat.shape[0])])
                    trajs = pred_gt_cat[:,1:]
                    traj_ends = trajs[:,-1,:].squeeze()
                    labels = KMeans(n_clusters=3, n_init='auto').fit_predict(traj_ends, sample_weight=probs)
                    reduced_traj = []
                    for k in range(3):
                        traj_k = trajs[labels == k]
                        prob_k = probs[labels == k]
                        reduced_traj.append((prob_k[:, np.newaxis, np.newaxis] * traj_k / prob_k.sum()).sum(axis=0))
                    reduced_traj = np.stack(reduced_traj, axis=0)
                    results.append(reduced_traj)

            for i, (scene_id, scene_primary_pedestrian_id, start_frame, end_frame, track_id,
                    pred_traj, gt_past) in enumerate(zip(scene_ids, scene_primary_pedestrian_ids, 
                                                start_frames, end_frames, track_ids, results, gt_pasts)):
                scene_data = dict()
                scene_data["id"] = scene_id
                scene_data["p"] = scene_primary_pedestrian_id
                scene_data["s"] = end_frame + 1
                scene_data["e"] = end_frame + 60
                scene_data["fps"] = 2
                scene = dict()
                scene["scene"] = scene_data
                writer.writerow(scene)
                
                preds = pred_traj.squeeze()
                vis_results.append(preds)
                vis_gt_pasts.append(gt_past)
                vis_pp_ids.append(scene_primary_pedestrian_id)
                K, L, D = preds.shape
                for k in range(K):
                    for l in range(L):
                        track_data = dict()
                        track_data["f"] = end_frame + l + 1
                        track_data["p"] = track_id.tolist()
                        track_data["x"] = preds[k,l,0]
                        track_data["y"] = preds[k,l,1]
                        track_data["prediction_number"] = k
                        track_data["scene_id"] = scene_id
                        track = dict()
                        track["track"] = track_data
                        writer.writerow(track)
        file.close()

        color_box = ['red', 'orange', 'yellow', 'green', 'blue', 'cyan', 'pink', 'purple', 'white', 'black']
        for i in range(len(vis_gt_pasts)):
            if i > 30:
                continue
            plt.plot(vis_gt_pasts[i][:,0], vis_gt_pasts[i][:,1], 
                     "-", linewidth=1.5, color='orange')
            for j in range(0,60):
                idx = j // 6
                c = color_box[idx]
                plt.scatter(vis_gt_pasts[i][j,0], vis_gt_pasts[i][j,1], linewidth=1.6, color=c)
        
            for j in range(len(vis_results[i])):
                plt.plot(vis_results[i][j,:,0], vis_results[i][j,:,1], 
                         "-", linewidth=1.5, color='g')
                plt.scatter(vis_results[i][j,-1,0], vis_results[i][j,-1,1], 
                            linewidth=1.6, color='r')
            save_path = "./vis/vis_test/" + str(f_idx)+"_"+str(vis_pp_ids[i]) + ".png"
            plt.savefig(save_path)
            plt.cla()
          
def gaussian_smoothing(data, sigma):
    smooth_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        smooth_data[:, i] = gaussian_filter1d(data[:, i], sigma=sigma, mode='nearest')
    return smooth_data
  
def moving_average(data, window_size):    
    left_window = window_size//10*9
    right_window = window_size//10
    smooth_data = np.zeros_like(data)
    
    for i in range(data.shape[1]):  # 对每一维度进行处理
        extended_data = np.pad(data[:, i], (left_window, right_window), mode='edge')
        cumsum = np.cumsum(extended_data)
        smooth_data[:, i] = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    
    return smooth_data
        
if __name__ == "__main__":
    main()
