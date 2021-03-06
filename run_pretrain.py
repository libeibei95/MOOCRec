"""
Script for pre-training the data

@author: Abinash Sinha
"""

from torch.utils.data import DataLoader, RandomSampler

import os
import argparse

from datasets import PretrainDataset
from trainers import PretrainTrainer
from models import EduRecModel

from utils import get_user_seqs_long_csv, check_path, set_seed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--plot_dir', default='plot', type=str)
    parser.add_argument('--output_dir', default='output', type=str)
    parser.add_argument('--data_name', default='MOOCCube', type=str)

    # model args
    parser.add_argument("--model_name", default='Pretrain', type=str)

    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=1, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)
    parser.add_argument('--num_intents', default=4, type=int)
    parser.add_argument('--lambda_', default=0.5, type=float)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=128, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    # pre train args
    parser.add_argument("--pre_epochs", type=int, default=300, help="number of pre_train epochs")
    parser.add_argument("--pre_batch_size", type=int, default=128)
    parser.add_argument('--ckp', default=20, type=int, help="pretrain epochs 10, 20, 30...")

    parser.add_argument("--mask_p", type=float, default=0.2, help="mask probability")
    parser.add_argument("--s2i_weight", type=float, default=1.0, help="seq2item loss weight")
    parser.add_argument("--s2s_weight", type=float, default=1.0, help="seq2seq loss weight")

    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    args = parser.parse_args()

    # set_seed(args.seed)
    # check_path(args.output_dir)
    #
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = os.path.join(args.data_dir, args.data_name + '.csv')

    user_seq, max_item, long_sequence = get_user_seqs_long_csv(args.data_file)
    # args.ckp = 20
    args_str = f'{args.model_name}-{args.data_name}-epochs-{args.ckp}'
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    # save model args
    args_str = f'{args.model_name}-{args.data_name}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    print(args)
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    model = EduRecModel(args=args)
    trainer = PretrainTrainer(model, None, None, None, args)

    # to resume training from last trained epoch
    if os.path.exists(args.checkpoint_path):
        trainer.load(args.checkpoint_path)
        print(f'Resume training from epoch={args.ckp} for pre-training!')
        init_epoch = int(args.ckp) - 1
    else:
        init_epoch = -1
    for epoch in range(args.pre_epochs):
        if epoch <= init_epoch:
          continue
        pretrain_dataset = PretrainDataset(args, user_seq, long_sequence)
        pretrain_sampler = RandomSampler(pretrain_dataset)
        pretrain_dataloader = DataLoader(pretrain_dataset, sampler=pretrain_sampler, batch_size=args.pre_batch_size)

        trainer.pretrain(epoch, pretrain_dataloader)

        # save checkpoint after execution of each epoch
        ckp = f'{args.model_name}-{args.data_name}-epochs-{epoch+1}.pt'
        checkpoint_path = os.path.join(args.output_dir, ckp)
        trainer.save(checkpoint_path)


main()
