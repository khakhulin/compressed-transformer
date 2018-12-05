import os
import sys


import time

import torch
import torch.nn as nn
from torchtext import data, datasets

import numpy as np
import dill as pickle

import nmt.transformer as transformer
import nmt.utils.optimizer as opt
import nmt.utils.train_utils as train_utils
from nmt.utils.arguments import init_config


from nmt.utils.prepare_data import prepare_data
import nmt.utils.train_utils as train_utils
import nmt.utils.optimizer as opt
import nmt.transformer as transformer

# TODO add logger!
# TODO add uniform initialization

def train(args):
    train_data, val_data, test_data, SRC, TGT = prepare_data(args)
        
    BATCH_SIZE = args.batch_size

    pad_idx = TGT.vocab.stoi["<pad>"]

    # TODO : add model parameters to config
    # TODO : add loading model
    print("Size of source vocabulary:", len(SRC.vocab))
    print("Size of target vocabulary:", len(TGT.vocab))

    print("FC matrix:", args.hidden_dim, args.ff_dim)
    model = transformer.make_model(len(SRC.vocab), len(TGT.vocab), 
                                   d_model=args.hidden_dim, d_ff=args.ff_dim, N=args.num_blocks, compress=args.compress)
    model.to(device)

    if args.load_model:
        print('load model from [%s]' % args.load_model, file=sys.stderr)
        params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
        # TODO args = params['args']
        state_dict = params['model']
        # opts = params['']
        model.load_state_dict(state_dict)


    criterion = train_utils.LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    # criterion = nn.NLLLoss(reduction="sum", ignore_index=0)
    criterion.to(device)
    train_iter = data.BucketIterator(train_data, batch_size=BATCH_SIZE, train=True, 
                                 sort_within_batch=True, 
                                 sort_key=lambda x: (len(x.src), len(x.trg)), repeat=False,
                                 device=device)
    valid_iter = data.Iterator(val_data, batch_size=BATCH_SIZE, train=False, sort=False, repeat=False,
                           device=device)

    model_opt = opt.WrapperOpt(model.src_embed[0].d_model, 1, 2000,
                                     torch.optim.Adam(model.parameters(), lr=args.lr))

    # train_time = begin_time = time.time()
    valid_params = (SRC, TGT, valid_iter)

    print("Number of examples in train: ", BATCH_SIZE * len([_ for _ in train_iter]))
    print("Number of examples in validation: ", BATCH_SIZE * len([_ for _ in valid_iter]))

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of parameters: ", params)

    if args.debug:
        model2 = transformer.make_model(len(SRC.vocab), len(TGT.vocab),
                                       d_model=args.hidden_dim, d_ff=args.ff_dim, N=args.num_blocks, compress=True)
        model_parameters2 = filter(lambda p: p.requires_grad, model2.parameters())
        params2 = sum([np.prod(p.size()) for p in model_parameters2])
        print("Number of parameters: ", params2)

        print("Tranable parameters ", params2)
        for name, param in model2.named_parameters():
            if param.requires_grad:
                print(name, param.data.size())

        print("compression rate: ", params/params2)

        exit()

    os.makedirs(os.path.dirname(args.save_to), exist_ok=True)

    for epoch in range(args.max_epoch):
        print("=" * 80)
        print("Epoch ", epoch + 1)
        print("=" * 80)
        print("Train...")
        model.train()
        train_utils.run_epoch(args, (train_utils.rebatch(pad_idx, b) for b in train_iter),
                              model,
                              train_utils.LossCompute(model.generator, criterion, model_opt),
                              valid_params=valid_params,
                              epoch_num = epoch)

        model.eval()
        print("Validation...")
        loss = train_utils.run_epoch(args, (train_utils.rebatch(pad_idx, b) for b in valid_iter), model,
                                     train_utils.LossCompute(model.generator, criterion, model_opt),
                                     valid_params=valid_params, is_valid=True)
        print()
        print("Validation perplexity ", np.exp(loss))


if __name__ == '__main__':
    args = init_config()
    print(args, file=sys.stderr)

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.mode == 'train':
        train(args)



