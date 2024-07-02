import numpy as np
import os
import sys
from fractions import gcd
from numbers import Number

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from utils import gpu, to_long,  Optimizer, StepLR

from layers import Conv1d, Res1d, Linear, LinearRes, Null, no_pad_Res1d, stride_Res1d
from numpy import float64, ndarray
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

file_path = os.path.abspath(__file__)
root_path = os.path.dirname(file_path)
model_name = os.path.basename(file_path).split(".")[0]

### config ###
config = dict()
"""Train"""
config["display_iters"] = 162676
config["val_iters"] = 162676
config["save_freq"] = 0.2
config["epoch"] = 0
config["horovod"] = True
config["opt"] = "adam"
config["num_epochs"] = 20
config["start_val_epoch"] = 0
config["lr"] = [5e-4, 1e-4]
config["lr_epochs"] = [10,]
config["lr_func"] = StepLR(config["lr"], config["lr_epochs"])

if "save_dir" not in config:
    config["save_dir"] = os.path.join(
        root_path, "best_model", model_name
    )

if not os.path.isabs(config["save_dir"]):
    config["save_dir"] = os.path.join(root_path, "best_model", config["save_dir"])

config["batch_size"] = 8
config["val_batch_size"] = 8
config["workers"] = 8
config["val_workers"] = config["workers"]

"""Dataset"""
root_path = "/mnt/home/data/giga2024/Trajectory/"
config["train_split"] = root_path + 'train/preprocess_end_stride_2/train_train.p'
config["val_split"] = root_path + 'train/preprocess_end_stride_2/train_val.p'
config["test_split"] = root_path + 'test/preprocess/test_1.p'

"""Model"""
config["rot_aug"] = False
config["n_actor"] = 64
config["actor2actor_dist"] = 25.0
config["pred_size"] = 1
config["pred_step"] = 1
config["num_preds"] = config["pred_size"] // config["pred_step"]
config["num_mods"] = 3
config["cls_coef"] = 1.0
config["end_coef"] = 1.
config["mgn"] = 0.2
config["cls_th"] = 2.0
config["cls_ignore"] = 0.2
### end of config ###

class Net(nn.Module):
    """
    Lane Graph Network contains following components:
        1. ActorNet: a 1D CNN to process the trajectory input
        2. MapNet: LaneGraphCNN to learn structured map representations 
           from vectorized map data
        3. Actor-Map Fusion Cycle: fuse the information between actor nodes 
           and lane nodes:
            a. A2M: introduces real-time traffic information to 
                lane nodes, such as blockage or usage of the lanes
            b. M2M:  updates lane node features by propagating the 
                traffic information over lane graphs
            c. M2A: fuses updated map features with real-time traffic 
                information back to actors
            d. A2A: handles the interaction between actors and produces
                the output actor features
        4. PredNet: prediction header for motion forecasting using 
           feature from A2A
    """
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config

        self.actor_net1 = ActorNet(config)
        self.actor_net2 = ActorNet(config)
        self.a2a = A2A(config)
        
        self.pred_net = PredNet(config)

    def forward(self, data: Dict) -> Dict[str, List[Tensor]]:
        actors, actor_idcs = actor_gather(gpu(data["feats"]))
        actor_ctrs = gpu(data["ctrs"])
        actors_1 = self.actor_net1(actors, actor_ctrs)
        actors_2 = self.actor_net2(actors, actor_ctrs)
        actors = actors_1 + actors_2
        actors = self.a2a(actors, actor_idcs, actor_ctrs)
        # prediction
        out = self.pred_net(actors, actor_idcs, actor_ctrs)
        rot, orig = gpu(data["rot"]), gpu(data["orig"])
        # transform prediction to world coordinates
        for i in range(len(out["reg"])):
            out["reg"][i] = torch.matmul(out["reg"][i], rot[i]) + orig[i].view(1, 1, 1, -1)
        return out
  
def actor_gather(actors: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
    batch_size = len(actors)
    num_actors = [len(x) for x in actors]

    actors = [x.transpose(1, 2) for x in actors]
    actors = torch.cat(actors, 0)

    actor_idcs = []
    count = 0
    for i in range(batch_size):
        idcs = torch.arange(count, count + num_actors[i]).to(actors.device)
        actor_idcs.append(idcs)
        count += num_actors[i]
    return actors, actor_idcs
      
class ActorNet(nn.Module):
    """
    Actor feature extractor with Conv1D
    """
    def __init__(self, config):
        super(ActorNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_in = 3
        n_out = [16, 32, 64]
        blocks = [Res1d, Res1d, Res1d]
        num_blocks = [2, 2, 2]

        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=ng))

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        n = config["n_actor"]
        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], n, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)
      
        ctrs_in = 2
        self.lstm_h0_init_function = nn.Linear(ctrs_in, n, bias=False)
        self.lstm_encoder = nn.LSTM(n, n, batch_first=True)
        self.output = Res1d(n, n, norm=norm, ng=ng)

    def forward(self, actors: Tensor, actor_ctrs) -> Tensor:
        actor_ctrs = torch.cat(actor_ctrs, 0)
        out = actors
        M,d,L = actors.shape
        outputs = []
        for i in range(len(self.groups)):
            out = self.groups[i](out)
            outputs.append(out)

        out = self.lateral[-1](outputs[-1])
        for i in range(len(outputs) - 2, -1, -1):
            out = F.interpolate(out, scale_factor=2, mode="linear", align_corners=False)
            out += self.lateral[i](outputs[i])

        out = self.output(out)
        out_init = out[:, :, -1]
        h0 = self.lstm_h0_init_function(actor_ctrs).view(1, M, config["n_actor"])
        c0 = self.lstm_h0_init_function(actor_ctrs).view(1, M, config["n_actor"])
        out = out.transpose(1, 2).contiguous()
        output, (hn, cn) = self.lstm_encoder(out, (h0, c0))
        out_lstm = hn.contiguous().view(M, config["n_actor"])
        out = out_lstm + out_init
      
        return out
      
class A2A(nn.Module):
    """
    The actor to actor block performs interactions among actors.
    """
    def __init__(self, config):
        super(A2A, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]

        att = []
        for i in range(2):
            att.append(Att(n_actor, n_actor))
        self.att = nn.ModuleList(att)

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor]) -> Tensor:
        for i in range(len(self.att)):
            actors = self.att[i](
                actors,
                actor_idcs,
                actor_ctrs,
                actors,
                actor_idcs,
                actor_ctrs,
                self.config["actor2actor_dist"],
            )
        return actors

class EncodeDist(nn.Module):
    def __init__(self, n, linear=True):
        super(EncodeDist, self).__init__()
        norm = "GN"
        ng = 1

        block = [nn.Linear(2, n), nn.ReLU(inplace=True)]

        if linear:
            block.append(nn.Linear(n, n))

        self.block = nn.Sequential(*block)

    def forward(self, dist):
        x, y = dist[:, :1], dist[:, 1:]
        dist = torch.cat(
            (
                torch.sign(x) * torch.log(torch.abs(x) + 1.0),
                torch.sign(y) * torch.log(torch.abs(y) + 1.0),
            ),
            1,
        )

        dist = self.block(dist)
        return dist
    
class PredNet(nn.Module):
    """
    Final motion forecasting with Linear Residual block
    """
    def __init__(self, config):
        super(PredNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]

        pred = []
        for i in range(config["num_mods"]):
            pred.append(
                nn.Sequential(
                    LinearRes(n_actor, n_actor, norm=norm, ng=ng),
                    nn.Linear(n_actor, 2 * config["num_preds"]),
                )
            )
        self.pred = nn.ModuleList(pred)

        self.att_dest = AttDest(n_actor)
        self.cls = nn.Sequential(
            LinearRes(n_actor, n_actor, norm=norm, ng=ng), nn.Linear(n_actor, 1)
        )
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor],
               ) -> Dict[str, List[Tensor]]:
        agent_index = torch.tensor([idcs[0] for idcs in actor_idcs])
        agent_ctrs = [actor_ctr[0:1] for actor_ctr in actor_ctrs]
        agents = actors[agent_index]
        preds = []
        for i in range(len(self.pred)):
            preds.append(self.pred[i](agents))
        reg = torch.cat([x.unsqueeze(1) for x in preds], 1)
        reg = reg.view(reg.size(0), reg.size(1), -1, 2)

        for i in range(len(actor_idcs)):
            ctrs = agent_ctrs[i].view(-1, 1, 1, 2)
            reg[i] = reg[i] + ctrs

        dest_ctrs = reg[:, :, -1].detach()
        feats = self.att_dest(agents, torch.cat(agent_ctrs, 0), dest_ctrs)
        cls = self.cls(feats).view(-1, self.config["num_mods"])
        #cls = self.softmax(cls)
        
        cls, sort_idcs = cls.sort(1, descending=True)
        row_idcs = torch.arange(len(sort_idcs)).long().to(sort_idcs.device)
        row_idcs = row_idcs.view(-1, 1).repeat(1, sort_idcs.size(1)).view(-1)
        sort_idcs = sort_idcs.view(-1)
        reg = reg[row_idcs, sort_idcs].view(cls.size(0), cls.size(1), -1, 2)

        out = dict()
        out["cls"], out["reg"] = [], []
        for i in range(len(actor_idcs)):
            out["cls"].append(cls[i])
            out["reg"].append(reg[i])
        return out

class Att(nn.Module):
    """
    Attention block to pass context nodes information to target nodes
    This is used in Actor2Map, Actor2Actor, Map2Actor and Map2Map
    """
    def __init__(self, n_agt: int, n_ctx: int) -> None:
        super(Att, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_ctx),
            nn.ReLU(inplace=True),
            Linear(n_ctx, n_ctx, norm=norm, ng=ng),
        )

        self.query = Linear(n_agt, n_ctx, norm=norm, ng=ng)

        self.ctx = nn.Sequential(
            Linear(3 * n_ctx, n_agt, norm=norm, ng=ng),
            nn.Linear(n_agt, n_agt, bias=False),
        )

        self.agt = nn.Linear(n_agt, n_agt, bias=False)
        self.norm = nn.GroupNorm(gcd(ng, n_agt), n_agt)
        self.linear = Linear(n_agt, n_agt, norm=norm, ng=ng, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, agts: Tensor, agt_idcs: List[Tensor], agt_ctrs: List[Tensor], ctx: Tensor, ctx_idcs: List[Tensor], ctx_ctrs: List[Tensor], dist_th: float) -> Tensor:
        res = agts
        if len(ctx) == 0:
            agts = self.agt(agts)
            agts = self.relu(agts)
            agts = self.linear(agts)
            agts += res
            agts = self.relu(agts)
            return agts

        batch_size = len(agt_idcs)
        hi, wi = [], []
        hi_count, wi_count = 0, 0
        for i in range(batch_size):
            dist = agt_ctrs[i].view(-1, 1, 2) - ctx_ctrs[i].view(1, -1, 2)
            dist = torch.sqrt((dist ** 2).sum(2))
            mask = dist <= dist_th

            idcs = torch.nonzero(mask, as_tuple=False)
            if len(idcs) == 0:
                continue

            hi.append(idcs[:, 0] + hi_count)
            wi.append(idcs[:, 1] + wi_count)
            hi_count += len(agt_idcs[i])
            wi_count += len(ctx_idcs[i])
        hi = torch.cat(hi, 0)
        wi = torch.cat(wi, 0)

        agt_ctrs = torch.cat(agt_ctrs, 0)
        ctx_ctrs = torch.cat(ctx_ctrs, 0)
        dist = agt_ctrs[hi] - ctx_ctrs[wi]
        dist = self.dist(dist)

        query = self.query(agts[hi])

        ctx = ctx[wi]
        ctx = torch.cat((dist, query, ctx), 1)
        ctx = self.ctx(ctx)

        agts = self.agt(agts)
        agts.index_add_(0, hi, ctx)
        agts = self.norm(agts)
        agts = self.relu(agts)

        agts = self.linear(agts)
        agts += res
        agts = self.relu(agts)
        return agts

class AttDest(nn.Module):
    def __init__(self, n_agt: int):
        super(AttDest, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_agt),
            nn.ReLU(inplace=True),
            Linear(n_agt, n_agt, norm=norm, ng=ng),
        )

        self.agt = Linear(2 * n_agt, n_agt, norm=norm, ng=ng)

    def forward(self, agts: Tensor, agt_ctrs: Tensor, dest_ctrs: Tensor) -> Tensor:
        n_agt = agts.size(1)
        num_mods = dest_ctrs.size(1)

        dist = (agt_ctrs.unsqueeze(1) - dest_ctrs).view(-1, 2)
        dist = self.dist(dist)
        agts = agts.unsqueeze(1).repeat(1, num_mods, 1).view(-1, n_agt)

        agts = torch.cat((dist, agts), 1)
        agts = self.agt(agts)
        return agts

class PredLoss(nn.Module):
    def __init__(self, config):
        super(PredLoss, self).__init__()
        self.config = config
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, out: Dict[str, List[Tensor]], gt_preds: List[Tensor], 
                has_preds: List[Tensor]) -> Dict[str, Union[Tensor, int]]:
        cls, reg = out["cls"], out["reg"]
        cls = torch.cat([x.unsqueeze(0) for x in cls], 0)
        reg = torch.cat([x for x in reg], 0).squeeze()
        
        gt_preds = torch.cat([x[0:1] for x in gt_preds], 0).squeeze()
        has_preds = torch.cat([x[0:1] for x in has_preds], 0)

        loss_out = dict()
        zero = 0.0 * (cls.sum() + reg.sum())
        loss_out["cls_loss"] = zero.clone()
        loss_out["num_cls"] = 0       
        loss_out["end_loss"] = zero.clone()
        loss_out["num_end"] = 0
        
        num_mods, num_preds = self.config["num_mods"], self.config["num_preds"]

        dist = []
        for j in range(num_mods):
            dist.append(
                torch.sqrt(
                    (
                        (reg[:, j] - gt_preds)
                        ** 2
                    ).sum(1)
                )
            )
        
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)
        min_dist, min_idcs = dist.min(1)
        row_idcs = torch.arange(len(min_idcs)).long().to(min_idcs.device)
        
        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.config["mgn"]
        
        coef = self.config["cls_coef"]
        loss_out["cls_loss"] += coef * (
            self.config["mgn"] * mask.sum() - mgn[mask].sum()
        )
        loss_out["num_cls"] += mask.sum().item()

        reg = reg[row_idcs, min_idcs]
        coef = self.config["end_coef"]
        loss_out["end_loss"] += coef * (self.reg_loss(reg[row_idcs], gt_preds[row_idcs]))
        loss_out["num_end"] += len(reg)  
        
        return loss_out

class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.pred_loss = PredLoss(config)

    def forward(self, out: Dict, data: Dict) -> Dict:
        loss_out = self.pred_loss(out, gpu(data["gt_preds"]), gpu(data["has_preds"]))
        loss_out["loss"] = loss_out["cls_loss"] / (
            loss_out["num_cls"] + 1e-10
        ) + loss_out["end_loss"] / (
            loss_out["num_end"] + 1e-10
        ) 
        return loss_out

class PostProcess(nn.Module):
    def __init__(self, config):
        super(PostProcess, self).__init__()
        self.config = config

    def forward(self, out,data):
        post_out = dict()
        post_out["preds"] = [x.detach().cpu().numpy() for x in out["reg"]]
        post_out["cls"] = [x.unsqueeze(0).detach().cpu().numpy() for x in out["cls"]]
        post_out["gt_preds"] = [x[0:1].numpy() for x in data["gt_preds"]]
        post_out["has_preds"] = [x[0:1] for x in data["has_preds"]]
        return post_out

    def append(self, metrics: Dict, loss_out: Dict, post_out: Optional[Dict[str, List[ndarray]]]=None) -> Dict:
        if len(metrics.keys()) == 0:
            for key in loss_out:
                if key != "loss":
                    metrics[key] = 0.0

            for key in post_out:
                metrics[key] = []

        for key in loss_out:
            if key == "loss":
                continue
            if isinstance(loss_out[key], torch.Tensor):
                metrics[key] += loss_out[key].item()
            else:
                metrics[key] += loss_out[key]

        for key in post_out:
            metrics[key] += post_out[key]
        return metrics

    def display(self, metrics, dt, epoch, lr=None):
        """Every display-iters print training/val information"""
        if lr is not None:
            print("Epoch %3.3f, lr %.5f, time %3.2f" % (epoch, lr, dt))
        else:
            print(
                "************************* Validation, time %3.2f *************************"
                % dt
            )

        cls = metrics["cls_loss"] / (metrics["num_cls"] + 1e-10)
        end = metrics["end_loss"] / (metrics["num_end"] + 1e-10)
        loss = cls + end

        preds = np.concatenate(metrics["preds"], 0)
        preds_cls = np.concatenate(metrics["cls"], 0)
        gt_preds = np.concatenate(metrics["gt_preds"], 0)
        has_preds = metrics["has_preds"]
        #has_preds = np.concatenate(metrics["has_preds"], 0)
        ade1, fde1, ade, fde, giga_score = pred_metrics(preds, gt_preds, has_preds, preds_cls)

        print(
            "loss %2.4f cls %2.4f, end %2.4f, ade1 %2.4f, fde1 %2.4f, ade %2.4f, fde %2.4f, giga_score %2.4f"
            % (loss, cls, end, ade1, fde1, ade, fde, giga_score)
        )
        print()


def pred_metrics(preds, gt_preds, has_preds, preds_cls):
    #assert has_preds.all()
    preds = np.asarray(preds, np.float32)
    gt_preds = np.asarray(gt_preds, np.float32)
    cls = np.asarray(preds_cls, np.float32)
    m, num_mods, num_preds, _ = preds.shape
    
    """batch_size x num_mods x num_preds"""
    err = np.sqrt(((preds - np.expand_dims(gt_preds, 1)) ** 2).sum(3))
    
    ade1 =  np.asarray([err[i, 0].mean() for i in range(m)]).mean()
    fde1 = err[row_idcs_last, 0].mean()
    #cls = softmax(cls, axis=1)
    min_idcs = err.argmin(1)
    row_idcs = np.arange(len(min_idcs)).astype(np.int64)
    err = err[row_idcs, min_idcs]
    cls = cls[row_idcs, min_idcs]
    ade = np.asarray([err[i].mean() for i in range(m)]).mean()
    fde = err.mean()
    giga_score = 0.7*ade + 0.3*fde
    return ade1, fde1, ade, fde, giga_score

class gigaDataset(Dataset):
    def __init__(self, split_file):
        self.split = np.load(split_file, allow_pickle=True)
            
    def __getitem__(self, idx):
        data = self.split[idx]
        return data
    
    def __len__(self):
        return len(self.split)
    
def collate_fn(batch):
    batch = from_numpy(batch)
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch

def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data

def get_model():
    net = Net(config)
    net = net.cuda()

    loss = Loss(config).cuda()
    post_process = PostProcess(config).cuda()

    params = net.parameters()
    opt = Optimizer(params, config)


    return config, gigaDataset, collate_fn, net, loss, post_process, opt
