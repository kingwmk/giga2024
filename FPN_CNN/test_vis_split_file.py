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

# define parser
parser = argparse.ArgumentParser(description="giga")
parser.add_argument(
    "-m1", "--model1", default="gigaNet_simple_aug", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight1", default="1.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)

root_path = "/mnt/home/data/giga2024/Trajectory/"
test_data_path = root_path + "test/preprocess_files/"

test_files_path = [test_data_path + 'test_1/',
                   test_data_path + 'test_2/',
                   test_data_path + 'test_3/', 
                   test_data_path + 'test_4/',
                   test_data_path + 'test_5/', 
                   test_data_path + 'test_6/', 
                   test_data_path + 'test_7/',
                   test_data_path + 'test_8/', 
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
    
    for f_idx in range(len(test_files)):
        vis_results, vis_assemble_results, vis_result_centers, vis_gt_pasts, vis_pp_ids = [], [], [], [], []
        #10 vis sample per test_file
        vis_count = 0
        file = open(output_files[f_idx], 'w')
        writer = ndjson.writer(file, ensure_ascii=False)
        # Data loader for evaluation
        test_split_files = [test_files_path[f_idx] + name for name in os.listdir(test_files_path[f_idx])]
        dataset = gigaDataset(test_split_files)
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

                output = net1(data)
                results = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output["reg"]]
                gt_pasts = [x[0].cpu().numpy().astype(np.float64) for x in data["origin_past_ctrs"]]
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
        
if __name__ == "__main__":
    main()

