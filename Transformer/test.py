from argparse import ArgumentParser

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

import GigaDataset
import GigaNet
root_path = "/mnt/home/data/giga2024/Trajectory/"
test_path = root_path + 'test/'
test_files_path = [test_path + 'transformer_preprocess_files/test_1/', 
                   test_path + 'transformer_preprocess_files/test_2/',
                   test_path + 'transformer_preprocess_files/test_3/', 
                   test_path + 'transformer_preprocess_files/test_4/',
                   test_path + 'transformer_preprocess_files/test_5/', 
                   test_path + 'transformer_preprocess_files/test_6/',
                   test_path + 'transformer_preprocess_files/test_7/', 
                   test_path + 'transformer_preprocess_files/test_8/',
             ]
if __name__ == '__main__':
    pl.seed_everything(2024, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default="GigaNet")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, required=True)
    args = parser.parse_args()

    model = {
        'GigaNet': GigaNet,
    }[args.model].load_from_checkpoint(checkpoint_path=args.ckpt_path)
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy='ddp')
    for i in range(len(test_files_path)):
        test_dataset = GigaDataset(processed_dir=test_files_path[i])
        dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
              num_workers=args.num_workers, pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
        trainer.test(model, dataloader)
