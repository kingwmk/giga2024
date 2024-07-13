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
    "-m1", "--model1", default="gigaNet_light_stride_4_test", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight1", default="0.902.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m2", "--model2", default="gigaNet_light_stride_4_test", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight2", default="1.504.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m3", "--model3", default="gigaNet_light_stride_4_test_2035", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight3", default="0.902.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m4", "--model4", default="gigaNet_light_stride_4_test_2046", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight4", default="0.902.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m5", "--model5", default="gigaNet_light_stride_4_test_2057", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight5", default="0.902.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m6", "--model6", default="gigaNet_light_stride_4_test_2068", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight6", default="0.902.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m7", "--model7", default="gigaNet_light_stride_4_test_2079", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight7", default="0.902.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m8", "--model8", default="gigaNet_light_stride_4_test_2080", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight8", default="0.902.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m9", "--model9", default="gigaNet_light_stride_4_test_2091", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight9", default="0.902.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m10", "--model10", default="gigaNet_light_stride_4_test_2002", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight10", default="0.902.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m11", "--model11", default="gigaNet_light_stride_4_test_2013", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight11", default="0.902.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)

"""
parser.add_argument(
    "-m12", "--model12", default="gigaNet_simple_12", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight12", default="20.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m13", "--model13", default="gigaNet_simple_13", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight13", default="20.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m14", "--model14", default="gigaNet_simple_14", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight14", default="20.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m15", "--model15", default="gigaNet_simple_15", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight15", default="20.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m16", "--model16", default="gigaNet_simple_16", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight16", default="20.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m17", "--model17", default="gigaNet_simple_17", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight17", default="20.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m18", "--model18", default="gigaNet_simple_18", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight18", default="20.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m19", "--model19", default="gigaNet_simple_19", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight19", default="20.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m20", "--model20", default="gigaNet_simple_20", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight20", default="20.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
"""
root_path = "/mnt/home/data/giga2024/Trajectory/"
test_data_path = root_path + "test/"

test_files = [test_data_path + 'preprocess/test_1.p',
              test_data_path + 'preprocess/test_2.p',
              test_data_path + 'preprocess/test_3.p', 
              test_data_path + 'preprocess/test_4.p',
              test_data_path + 'preprocess/test_5.p', 
              test_data_path + 'preprocess/test_6.p', 
              test_data_path + 'preprocess/test_7.p',
              test_data_path + 'preprocess/test_8.p', 
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

    #model7
    model7 = import_module(args.model7)
    config7, _, collate_fn7, net7, loss7, post_process7, opt7 = model7.get_model()

    # load pretrain model
    ckpt_path7 = args.weight7
    if not os.path.isabs(ckpt_path7):
        ckpt_path7 = os.path.join(config7["save_dir"], ckpt_path7)
    ckpt7 = torch.load(ckpt_path7, map_location=lambda storage, loc: storage)
    load_pretrain(net7, ckpt7["state_dict"])
    net7.eval()

    #model8
    model8 = import_module(args.model8)
    config8, _, collate_fn8, net8, loss8, post_process8, opt8 = model8.get_model()

    # load pretrain model
    ckpt_path8 = args.weight8
    if not os.path.isabs(ckpt_path8):
        ckpt_path8 = os.path.join(config8["save_dir"], ckpt_path8)
    ckpt8 = torch.load(ckpt_path8, map_location=lambda storage, loc: storage)
    load_pretrain(net8, ckpt8["state_dict"])
    net8.eval()
    
    #model9
    model9 = import_module(args.model9)
    config9, _, collate_fn9, net9, loss9, post_process9, opt9 = model9.get_model()

    # load pretrain model
    ckpt_path9 = args.weight9
    if not os.path.isabs(ckpt_path9):
        ckpt_path9 = os.path.join(config9["save_dir"], ckpt_path9)
    ckpt9 = torch.load(ckpt_path9, map_location=lambda storage, loc: storage)
    load_pretrain(net9, ckpt9["state_dict"])
    net9.eval()
    
    #model10
    model10 = import_module(args.model10)
    config10, _, collate_fn10, net10, loss10, post_process10, opt10 = model10.get_model()

    # load pretrain model
    ckpt_path10 = args.weight10
    if not os.path.isabs(ckpt_path10):
        ckpt_path10 = os.path.join(config10["save_dir"], ckpt_path10)
    ckpt10 = torch.load(ckpt_path10, map_location=lambda storage, loc: storage)
    load_pretrain(net10, ckpt10["state_dict"])
    net10.eval()
    
    #model11
    model11 = import_module(args.model11)
    config11, _, collate_fn11, net11, loss11, post_process11, opt11 = model11.get_model()

    # load pretrain model
    ckpt_path11 = args.weight11
    if not os.path.isabs(ckpt_path11):
        ckpt_path11 = os.path.join(config11["save_dir"], ckpt_path11)
    ckpt11 = torch.load(ckpt_path11, map_location=lambda storage, loc: storage)
    load_pretrain(net11, ckpt11["state_dict"])
    net11.eval()
    
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

                output = net1(data)
                results = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output["reg"]]
                output2 = net2(data)
                results2 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output2["reg"]]
                output3 = net3(data)
                results3 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output3["reg"]]
                output4 = net4(data)
                results4 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output4["reg"]]
                output5 = net5(data)
                results5 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output5["reg"]]
                output6 = net6(data)
                results6 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output6["reg"]]
                output7 = net7(data)
                results7 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output7["reg"]]
                output8 = net8(data)
                results8 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output8["reg"]]
                output9 = net9(data)
                results9 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output9["reg"]]
                output10 = net10(data)
                results10 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output10["reg"]]  
                output11 = net11(data)
                results11 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output11["reg"]]
                output12 = net12(data)
                results12 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output12["reg"]]
                output13 = net13(data)
                results13 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output13["reg"]]
                output14 = net14(data)
                results14 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output14["reg"]]
                output15 = net15(data)
                results15 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output15["reg"]]
                output16 = net16(data)
                results16 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output16["reg"]]
                output17 = net17(data)
                results17 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output17["reg"]]
                output18 = net18(data)
                results18 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output18["reg"]]
                output19 = net19(data)
                results19 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output19["reg"]]
                output20 = net20(data)
                results20 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output20["reg"]]
                
                gt_pasts = [x[0].cpu().numpy().astype(np.float64) for x in data["origin_past_ctrs"]]
                for i in range(len(results1)):
                    trajs = np.concatenate((results1[i],results2[i],results3[i],results4[i],
                                            results5[i],results6[i],results7[i],results8[i],
                                           results9[i],results10[i],results11[i],results12[i],
                                            results13[i],results14[i],results15[i],results16[i],
                                           results17[i],results18[i],results19[i],results20[i]), 1).squeeze()
                    traj_ends = trajs[:,-2,:].squeeze()
                    kmeans = KMeans(n_clusters=3, random_state=0).fit(traj_ends)
                    cts = kmeans.cluster_centers_
                    result = np.zeros((3, 60, 2), np.float64)
                    orign = gt_past[59:60]
                    for j in range(3):  
                        vel = (cts[j] - orign)/60
                        vel_pred = np.repeat(vel, 60, axis=0)
                    results.append(result)
                    vis_result_centers.append(kmeans.cluster_centers_)
                    vis_assemble_results.append(result)
                    vis_count = vis_count + 1
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
        
if __name__ == "__main__":
    main()
