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
import pickle

# define parser
parser = argparse.ArgumentParser(description="giga")
parser.add_argument(
    "-m1", "--model1", default="gigaNet_simple_1", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight1", default="20.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m2", "--model2", default="gigaNet_simple_2", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight2", default="20.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m3", "--model3", default="gigaNet_simple_3", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight3", default="20.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m4", "--model4", default="gigaNet_simple_4", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight4", default="20.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m5", "--model5", default="gigaNet_simple_5", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight5", default="20.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m6", "--model6", default="gigaNet_simple_6", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight6", default="20.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m7", "--model7", default="gigaNet_simple_7", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight7", default="20.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m8", "--model8", default="gigaNet_simple_8", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight8", default="20.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m9", "--model9", default="gigaNet_simple_9", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight9", default="20.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m10", "--model10", default="gigaNet_simple_10", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight10", default="20.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "-m11", "--model11", default="gigaNet_simple_11", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "--weight11", default="20.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
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

root_path = "/mnt/home/code/giga-trajectory-main/"
data_path = root_path + "dataset/"

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
    
output_files = ['./submission/results/test_1.ndjson',
                './submission/results/test_2.ndjson',
                './submission/results/test_3.ndjson', 
                './submission/results/test_4.ndjson',
                './submission/results/test_5.ndjson', 
                './submission/results/test_6.ndjson',
                './submission/results/test_7.ndjson', 
                './submission/results/test_8.ndjson',
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
    
    #model12
    model12 = import_module(args.model12)
    config12, _, collate_fn12, net12, loss12, post_process12, opt12 = model12.get_model()

    # load pretrain model
    ckpt_path12 = args.weight12
    if not os.path.isabs(ckpt_path12):
        ckpt_path12 = os.path.join(config12["save_dir"], ckpt_path12)
    ckpt12 = torch.load(ckpt_path12, map_location=lambda storage, loc: storage)
    load_pretrain(net12, ckpt12["state_dict"])
    net12.eval()
    
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
    """
    #model14
    model14 = import_module(args.model14)
    config14, _, collate_fn14, net14, loss14, post_process14, opt14 = model14.get_model()

    # load pretrain model
    ckpt_path14 = args.weight14
    if not os.path.isabs(ckpt_path14):
        ckpt_path14 = os.path.join(config14["save_dir"], ckpt_path14)
    ckpt14 = torch.load(ckpt_path14, map_location=lambda storage, loc: storage)
    load_pretrain(net14, ckpt14["state_dict"])
    net14.eval()
    
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
    
    #model6
    model16 = import_module(args.model16)
    config16, _, collate_fn16, net16, loss16, post_process16, opt16 = model16.get_model()

    # load pretrain model
    ckpt_path16 = args.weight16
    if not os.path.isabs(ckpt_path16):
        ckpt_path16 = os.path.join(config16["save_dir"], ckpt_path16)
    ckpt16 = torch.load(ckpt_path16, map_location=lambda storage, loc: storage)
    load_pretrain(net16, ckpt16["state_dict"])
    net16.eval()   
    
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
    
    #model18
    model18 = import_module(args.model18)
    config18, _, collate_fn18, net18, loss18, post_process18, opt18 = model18.get_model()

    # load pretrain model
    ckpt_path18 = args.weight18
    if not os.path.isabs(ckpt_path18):
        ckpt_path18 = os.path.join(config18["save_dir"], ckpt_path18)
    ckpt18 = torch.load(ckpt_path18, map_location=lambda storage, loc: storage)
    load_pretrain(net18, ckpt18["state_dict"])
    net18.eval()   
    
    #model19
    model19 = import_module(args.model19)
    config19, _, collate_fn19, net19, loss19, post_process19, opt19 = model19.get_model()

    # load pretrain model
    ckpt_path19 = args.weight19
    if not os.path.isabs(ckpt_path19):
        ckpt_path19 = os.path.join(config19["save_dir"], ckpt_path19)
    ckpt19 = torch.load(ckpt_path19, map_location=lambda storage, loc: storage)
    load_pretrain(net19, ckpt19["state_dict"])
    net19.eval()
    
    #model20
    model20 = import_module(args.model20)
    config20, _, collate_fn20, net20, loss20, post_process20, opt20 = model20.get_model()

    # load pretrain model
    ckpt_path20 = args.weight20
    if not os.path.isabs(ckpt_path20):
        ckpt_path20 = os.path.join(config20["save_dir"], ckpt_path20)
    ckpt20 = torch.load(ckpt_path20, map_location=lambda storage, loc: storage)
    load_pretrain(net20, ckpt20["state_dict"])
    net20.eval()
    """
    for f_idx in range(len(test_files)):
        test_prediction = []
        for n in range(13):
            test_prediction.append({})
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
                """
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
                    sid = scene_ids[i]
                    pid = scene_primary_pedestrian_ids[i]
                    sf = end_frames[i] + 1
                    ef = end_frames[i] + 60
                    test_prediction[0][sid] = (results1[i].squeeze(), cls1[i], sid, pid, sf, ef)
                    test_prediction[1][sid] = (results2[i].squeeze(), cls2[i], sid, pid, sf, ef)
                    test_prediction[2][sid] = (results3[i].squeeze(), cls3[i], sid, pid, sf, ef)
                    test_prediction[3][sid] = (results4[i].squeeze(), cls4[i], sid, pid, sf, ef)
                    test_prediction[4][sid] = (results5[i].squeeze(), cls5[i], sid, pid, sf, ef)
                    test_prediction[5][sid] = (results6[i].squeeze(), cls6[i], sid, pid, sf, ef)
                    test_prediction[6][sid] = (results7[i].squeeze(), cls7[i], sid, pid, sf, ef)
                    test_prediction[7][sid] = (results8[i].squeeze(), cls8[i], sid, pid, sf, ef)
                    test_prediction[8][sid] = (results9[i].squeeze(), cls9[i], sid, pid, sf, ef)
                    test_prediction[9][sid] = (results10[i].squeeze(), cls10[i], sid, pid, sf, ef)
                    test_prediction[10][sid] = (results11[i].squeeze(), cls11[i], sid, pid, sf, ef)
                    test_prediction[11][sid] = (results12[i].squeeze(), cls12[i], sid, pid, sf, ef)
                    test_prediction[12][sid] = (results13[i].squeeze(), cls13[i], sid, pid, sf, ef)
                    
                    trajs = np.concatenate((results1[i],results2[i],results3[i],results4[i],
                                            results5[i],results6[i],results7[i],results8[i],
                                            results9[i],results10[i],results11[i],results12[i],
                                            results13[i],
                                            #results14[i],results15[i],results16[i],
                                            #results17[i],results18[i],results19[i],results20[i]
                                           ), 1).squeeze()
                    probs = np.concatenate((cls1[i],cls2[i],cls3[i],cls4[i],
                                            cls5[i],cls6[i],cls7[i],cls8[i],
                                            cls9[i],cls10[i],cls11[i],cls12[i],
                                            cls13[i],
                                            #cls14[i],cls15[i],cls16[i],
                                            #cls17[i],cls18[i],cls19[i],cls20[i]
                                           ), 0).squeeze()
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
                orign = gt_past[59:60]
                if (gt_past[59:60][0]!=0 or gt_past[59:60][1]!=0) and (
                    gt_past[58:59][0]!=0 or gt_past[58:59][1]!=0):
                    vel = gt_past[59:60] - gt_past[58:59]
                    vel_pred = np.repeat(vel, 60, axis=0)
                    vel_pred = orign + vel_pred.cumsum(0)
                    min_idx = 0
                    min_dis = 9999999
                    for k in range(3):
                        dis = math.dist(preds[k][-1], vel_pred[-1])
                        if dis < min_dis:
                            min_dis = dis
                            min_idx = k
                    preds[min_idx] = vel_pred
                                  
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
        for i in range(13):
            file_path = './submission/test_' + str(f_idx + 1) + '_' + str(i) + '_' + '20.pkl'
            file = open(file_path, 'wb')
            pickle.dump(test_prediction[i], file)
            file.close()
        
        """
        color_box = ['red','orange','yellow','green','blue','cyan','pink','purple','black']
        for i in range(len(vis_gt_pasts)):
            plt.plot(vis_gt_pasts[i][:,0], vis_gt_pasts[i][:,1], 
                     "-", linewidth=1.5, color='orange')
            for j in range(9):
                plt.scatter(vis_gt_pasts[i][j,0], vis_gt_pasts[i][j,1], linewidth=1.6, color=color_box[j])
        
            for j in range(len(vis_assemble_results[i])):
                plt.plot(vis_assemble_results[i][j,:,0], vis_assemble_results[i][j,:,1], 
                         "-", linewidth=1.5, color='g')
                plt.scatter(vis_result_centers[i][j][0], vis_result_centers[i][j][1], 
                            linewidth=1.5, color='r')
            save_path = data_path + "vis/vis_test_assemble_preds/" + str(f_idx)+"_"+str(vis_pp_ids[i]) + ".png"
            plt.savefig(save_path)
            plt.cla()
        """
        
if __name__ == "__main__":
    main()
