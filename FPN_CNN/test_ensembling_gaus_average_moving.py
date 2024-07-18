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
    for f_idx in range(len(test_files)):
        vis_results, vis_assemble_results, vis_result_centers, vis_gt_pasts, vis_pp_ids = [], [], [], [], []
        #10 vis sample per test_file
        vis_count = 0
        file = open(output_files[f_idx], 'w')
        writer = ndjson.writer(file, ensure_ascii=False)
        raw_result_file_path = "./raw_results/" + str(f_idx) + ".pkl"
        raw_results = pickle.load(raw_result_file_path)
        # begin inference
        for ii in tqdm(range(len(raw_results[1]))):
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
                output7 = net7(data)
                results7 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output7["reg"]]
                cls7 = [x.detach().cpu().numpy().astype(np.float64) for x in output7["cls"]]
                cls7 = softmax(cls7, axis=-1)
                output8 = net8(data)
                results8 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output8["reg"]]
                cls8 = [x.detach().cpu().numpy().astype(np.float64) for x in output8["cls"]]
                cls8 = softmax(cls8, axis=-1)
                output9 = net9(data)
                results9 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output9["reg"]]
                cls9 = [x.detach().cpu().numpy().astype(np.float64) for x in output9["cls"]]
                cls9 = softmax(cls9, axis=-1)
                output10 = net10(data)
                results10 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output10["reg"]]  
                cls10 = [x.detach().cpu().numpy().astype(np.float64) for x in output10["cls"]]
                cls10 = softmax(cls10, axis=-1)
                output11 = net11(data)
                results11 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output11["reg"]]
                cls11 = [x.detach().cpu().numpy().astype(np.float64) for x in output11["cls"]]
                cls11 = softmax(cls11, axis=-1)
                output12 = net12(data)
                results12 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output12["reg"]]
                cls12 = [x.detach().cpu().numpy().astype(np.float64) for x in output12["cls"]]
                cls12 = softmax(cls12, axis=-1)
                output13 = net13(data)
                results13 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output13["reg"]]
                cls13 = [x.detach().cpu().numpy().astype(np.float64) for x in output13["cls"]]
                cls13 = softmax(cls13, axis=-1)
                output14 = net14(data)
                results14 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output14["reg"]]
                cls14 = [x.detach().cpu().numpy().astype(np.float64) for x in output14["cls"]]
                cls14 = softmax(cls14, axis=-1)
                output15 = net15(data)
                results15 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output15["reg"]]
                cls15 = [x.detach().cpu().numpy().astype(np.float64) for x in output15["cls"]]
                cls15 = softmax(cls15, axis=-1)
                output16 = net16(data)
                results16 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output16["reg"]]
                cls16 = [x.detach().cpu().numpy().astype(np.float64) for x in output16["cls"]]
                cls16 = softmax(cls16, axis=-1)
                output17 = net17(data)
                results17 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output17["reg"]]
                cls17 = [x.detach().cpu().numpy().astype(np.float64) for x in output17["cls"]]
                cls17 = softmax(cls17, axis=-1)
                output18 = net18(data)
                results18 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output18["reg"]]
                cls18 = [x.detach().cpu().numpy().astype(np.float64) for x in output18["cls"]]
                cls18 = softmax(cls18, axis=-1)
                """
                output19 = net19(data)
                results19 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output19["reg"]]
                cls19 = [x.detach().cpu().numpy().astype(np.float64) for x in output19["cls"]]
                cls19 = softmax(cls19, axis=-1)
                output20 = net20(data)
                results20 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output20["reg"]] 
                cls20 = [x.detach().cpu().numpy().astype(np.float64) for x in output20["cls"]]
                cls20 = softmax(cls20, axis=-1)
                """
                results = []
                gt_pasts = [x[0].cpu().numpy().astype(np.float64) for x in data["origin_past_ctrs"]]
                for i in range(len(results1)):
                    trajs = np.concatenate((results1[i],results2[i],results3[i],results4[i],
                                            results5[i],results6[i],results7[i],results8[i],
                                            results9[i],results10[i],results11[i],results12[i],
                                            results13[i], results14[i],results15[i],results16[i],
                                            results17[i],results18[i],#results19[i],results20[i]
                                           ), 1).squeeze()
                    probs = np.concatenate((cls1[i],cls2[i],cls3[i],cls4[i],
                                            cls5[i],cls6[i],cls7[i],cls8[i],
                                            cls9[i],cls10[i],cls11[i],cls12[i],
                                            cls13[i], cls14[i],cls15[i],cls16[i],
                                            cls17[i],cls18[i],#cls19[i],cls20[i]
                                           ), 0).squeeze()
                    sigma = 20
                    trajs = np.array([gaussian_smoothing(trajs[s], sigma) for s in range(trajs.shape[0])])
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
        
if __name__ == "__main__":
    main()
