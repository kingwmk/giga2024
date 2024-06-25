# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import shutil
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from datamodules import ArgoverseV2DataModule
from datamodules import ArgoverseV1DataModule
from predictors import QCNet

if __name__ == '__main__':
    pl.seed_everything(2024, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='/mnt/home/data/argoverse2/motion')
    parser.add_argument('--resume', default="", type=str, metavar="RESUME", help="checkpoint path")
    parser.add_argument('--train_batch_size', type=int, default=5)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=7)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--test_raw_dir', type=str, default=None)
    parser.add_argument('--train_processed_dir', type=str, default=None)
    parser.add_argument('--val_processed_dir', type=str, default=None)
    parser.add_argument('--test_processed_dir', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, default=7)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--log_path', type=str, default='./log/test')
    QCNet.add_model_specific_args(parser)
    args = parser.parse_args()
    model = QCNet(**vars(args))
    src_dirs = ["./", "./modules", "./predictors", "./datasets"]
    dst_dirs = [args.log_path, args.log_path+"/modules", args.log_path+"/predictors", args.log_path+"/datasets"]
    for src_dir, dst_dir in zip(src_dirs, dst_dirs):
        files = [f for f in os.listdir(src_dir) if f.endswith(".py")]
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for f in files:
            shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))
            
    datamodule = {
        'argoverse_v2': ArgoverseV2DataModule,
        'argoverse_v1': ArgoverseV1DataModule,
    }[args.dataset](**vars(args))
    model_checkpoint = ModelCheckpoint(monitor='val_minFDE', save_top_k=6, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,
            strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True),
            callbacks=[model_checkpoint, lr_monitor], max_epochs=args.max_epochs, default_root_dir=args.log_path)
    if args.resume:
        trainer.fit(model, datamodule, ckpt_path=args.resume)
    else:
        trainer.fit(model, datamodule)
