import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
    "--weight1", default="1.804.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)

parser.add_argument(
    "-m3", "--model3", default="gigaNet_light_stride_4_test_2035", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight3", default="1.804.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)

parser.add_argument(
    "-m5", "--model5", default="gigaNet_light_stride_4_test_2046", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight5", default="1.804.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)

parser.add_argument(
    "-m7", "--model7", default="gigaNet_light_stride_4_test_2057", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight7", default="1.804.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)

parser.add_argument(
    "-m9", "--model9", default="gigaNet_light_stride_4_test_2068", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight9", default="1.804.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)

parser.add_argument(
    "-m11", "--model11", default="gigaNet_light_stride_4_test_2079", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight11", default="1.804.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)

parser.add_argument(
    "-m13", "--model13", default="gigaNet_light_stride_4_test_2080", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight13", default="1.804.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)

parser.add_argument(
    "-m15", "--model15", default="gigaNet_light_stride_4_test_2091", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight15", default="1.804.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)

parser.add_argument(
    "-m17", "--model17", default="gigaNet_light_stride_4_test_2002", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight17", default="1.804.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)

parser.add_argument(
    "-m19", "--model19", default="gigaNet_light_stride_4_test_2013", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight19", default="1.804.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
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
    
output_files = ['submission/1804/test_1.ndjson',
                'submission/1804/test_2.ndjson',
                'submission/1804/test_3.ndjson', 
                'submission/1804/test_4.ndjson',
                'submission/1804/test_5.ndjson', 
                'submission/1804/test_6.ndjson',
                'submission/1804/test_7.ndjson', 
                'submission/1804/test_8.ndjson',
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

    #model13
    model13 = import_module(args.model13)
    config13, _, collate_fn13, net13, loss13, post_process13, opt13 = model13.get_model()

    # load pretrain model
    ckpt_path13 = args.weight13
    if not os.path.isabs(ckpt_path13):
        ckpt_path13 = os.path.join(config13["save_dir"], ckpt_path13)
    ckpt13 = torch.load(ckpt_path13, map_location=lambda storage, loc: storage)
    load_pretrain(net13, ckpt13["state_dict"])
    net13.eval()
  
    #model5
    model15 = import_module(args.model15)
    config15, _, collate_fn15, net15, loss15, post_process15, opt15 = model15.get_model()

    # load pretrain model
    ckpt_path15 = args.weight15
    if not os.path.isabs(ckpt_path15):
        ckpt_path15 = os.path.join(config15["save_dir"], ckpt_path15)
    ckpt15 = torch.load(ckpt_path15, map_location=lambda storage, loc: storage)
    load_pretrain(net15, ckpt15["state_dict"])
    net15.eval()
    
    #model17
    model17 = import_module(args.model17)
    config17, _, collate_fn17, net17, loss17, post_process17, opt17 = model17.get_model()

    # load pretrain model
    ckpt_path17 = args.weight17
    if not os.path.isabs(ckpt_path17):
        ckpt_path17 = os.path.join(config17["save_dir"], ckpt_path17)
    ckpt17 = torch.load(ckpt_path17, map_location=lambda storage, loc: storage)
    load_pretrain(net17, ckpt17["state_dict"])
    net17.eval()

    model19 = import_module(args.model19)
    config19, _, collate_fn19, net19, loss19, post_process19, opt19 = model19.get_model()
    # load pretrain model
    ckpt_path19 = args.weight19
    if not os.path.isabs(ckpt_path19):
        ckpt_path19 = os.path.join(config19["save_dir"], ckpt_path19)
    ckpt19 = torch.load(ckpt_path19, map_location=lambda storage, loc: storage)
    load_pretrain(net19, ckpt19["state_dict"])
    net19.eval()
  
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
                results1 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output["reg"]]
                output3 = net3(data)
                results3 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output3["reg"]]
                
                output5 = net5(data)
                results5 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output5["reg"]]
               
                output7 = net7(data)
                results7 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output7["reg"]]
                
                output9 = net9(data)
                results9 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output9["reg"]]
                
                output11 = net11(data)
                results11 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output11["reg"]]
               
                output13 = net13(data)
                results13 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output13["reg"]]
                
                output15 = net15(data)
                results15 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output15["reg"]]
                
                output17 = net17(data)
                results17 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output17["reg"]]
                output19 = net19(data)
                results19 = [x[0:1,:3].detach().cpu().numpy().astype(np.float64) for x in output19["reg"]]                
                
                gt_pasts = [x[0].cpu().numpy().astype(np.float64) for x in data["origin_past_ctrs"]]
                results = []
                for i in range(len(results1)):
                    trajs = np.concatenate((results1[i],results3[i],
                                            results5[i],results7[i],
                                           results9[i],results11[i],
                                            results13[i],results15[i],
                                           results17[i],results19[i]), 1).squeeze()
                    traj_ends = trajs[:,-1,:].squeeze()
                    kmeans = KMeans(n_clusters=3, n_init='auto', random_state=0).fit(traj_ends)
                    cts = kmeans.cluster_centers_
                    result = np.zeros((3, 60, 2), np.float64)
                    orign = gt_pasts[i][59:60]
                    for j in range(3):  
                        vel = (cts[j] - orign)/60
                        vel_pred = np.repeat(vel, 60, axis=0)
                        result[j] = orign + vel_pred.cumsum(0)
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
