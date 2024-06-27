import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import shutil
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from datamodules import GigaDataModule
import GigaNet

if __name__ == '__main__':
    pl.seed_everything(2024, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--resume', default="", type=str, metavar="RESUME", help="checkpoint path")
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--train_processed_dir', type=str, default=None)
    parser.add_argument('--val_processed_dir', type=str, default=None)
    parser.add_argument('--test_processed_dir', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, default=8)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--log_path', type=str, default='./log/test')
    GigaNet.add_model_specific_args(parser)
    args = parser.parse_args()
    model = GigaNet(**vars(args))
    src_dirs = ["./",]
    dst_dirs = [args.log_path,]
    for src_dir, dst_dir in zip(src_dirs, dst_dirs):
        files = [f for f in os.listdir(src_dir) if f.endswith(".py")]
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for f in files:
            shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))
            
    datamodule = GigaDataModule(**vars(args))
    model_checkpoint = ModelCheckpoint(monitor='val_minFDE', save_top_k=20, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,
            strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True),
            callbacks=[model_checkpoint, lr_monitor], max_epochs=args.max_epochs, default_root_dir=args.log_path)
    if args.resume:
        trainer.fit(model, datamodule, ckpt_path=args.resume)
    else:
        trainer.fit(model, datamodule)
