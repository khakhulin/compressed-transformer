import os
import sys


import time

import torch
import torch.nn as nn
from torchtext import data, datasets

import numpy as np
import dill as pickle
import random
import nmt.transformer as transformer
import nmt.utils.optimizer as opt
import nmt.utils.train_utils as train_utils
from nmt.utils.arguments import init_config
from nmt.utils.gpu_utils import MultiGPULossCompute

from nmt.utils.prepare_data import prepare_data
import nmt.utils.train_utils as train_utils
import nmt.utils.optimizer as opt
import nmt.transformer as transformer

# TODO add logger!
# TODO add uniform initialization


def debug_compress_info(model, model2):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    model_parameters2 = filter(lambda p: p.requires_grad, model2.parameters())
    params2 = sum([np.prod(p.size()) for p in model_parameters2])
    print("Number of parameters in original model: ", params)

    print("Number of parameters in compress model: ", params2)

    num_embd1 = []
    num_embd2 = []
    w1_param = []

    flag = False

    for name, param in model2.named_parameters():
        if name.__contains__("decoder.layers.5.feed_forward.w_1"):
            if not flag:
                print(param.size())
                flag = True
            w1_param.append(np.prod(param.size()))
        if name.__contains__("src_embed"):
            num_embd2.append(np.prod(param.size()))
        if name.__contains__("tgt_embed"):
            num_embd2.append(np.prod(param.size()))
        # print(name, param.data.size())

    print("Num parameters in compress fc layer", np.sum(w1_param))
    flag = False

    w1_param = []
    for name, param in model.named_parameters():
        if name.__contains__("decoder.layers.5.feed_forward.w_1"):
            w1_param.append(np.prod(param.size()))
            if not flag:
                print(param.size())
                flag = True
        if name.__contains__("src_embed"):
            num_embd1.append(np.prod(param.size()))
        if name.__contains__("tgt_embed"):
            num_embd1.append(np.prod(param.size()))

    print("Num parameters in original fc layer", np.sum(w1_param))

    print("Number of parameters in embeddings layer")
    print(np.sum(num_embd1))
    print(np.sum(num_embd2))

    print("compression rate: ", params / params2)

    print("compression rate without embeddings: ", (params - np.sum(num_embd1)) / (params2 - np.sum(num_embd2)))

    print()


def train(args):
    train_data, val_data, test_data, SRC, TGT = prepare_data(args)

    BATCH_SIZE = args.batch_size
    best_bleu_loss = 0
    pad_idx = TGT.vocab.stoi["<pad>"]

    # TODO : add model parameters to config
    # TODO : add loading model
    print("Size of source vocabulary:", len(SRC.vocab))
    print("Size of target vocabulary:", len(TGT.vocab))

    print("FC matrix:", args.hidden_dim, args.ff_dim)
    print(args.compress)
    model = transformer.make_model(len(SRC.vocab), len(TGT.vocab),
                                   d_model=args.hidden_dim, d_ff=args.ff_dim,
                                   N=args.num_blocks, compress=args.compress,
                                   compress_mode=args.compress_mode,
                                   num_compress_enc=args.num_enc_blocks_comp,
                                   num_compress_dec=args.num_dec_blocks_comp
                                   )
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
                                d_model=args.hidden_dim, d_ff=args.ff_dim,
                                N=args.num_blocks, compress=True,
                                num_compress_enc=args.num_enc_blocks_comp,
                                num_compress_dec=args.num_dec_blocks_comp)


        # print("Tranable parameters in fc module ", params2)
        debug_compress_info(model, model2)

        exit()

    os.makedirs(os.path.dirname(args.save_to), exist_ok=True)

    if args.multi_gpu:
        devices = list(np.arange(args.num_devices))
        model_parallel = nn.DataParallel(model, device_ids=devices)

    for epoch in range(args.max_epoch):
        print("=" * 80)
        print("Epoch ", epoch + 1)
        print("=" * 80)
        print("Train...")
        if args.multi_gpu:
            model_parallel.train()
            train_loss_fn = MultiGPULossCompute(model.generator, criterion,
                                                      devices=devices, opt=model_opt)
        else:
            train_loss_fn = train_utils.LossCompute(model.generator, criterion, model_opt)

            model.train()

        train_utils.run_epoch(args, (train_utils.rebatch(pad_idx, b) for b in train_iter),
                                  model_parallel, train_loss_fn,
                                  valid_params=valid_params,
                                  epoch_num=epoch)

        if args.multi_gpu:
            model_parallel.eval()
            val_loss_fn = MultiGPULossCompute(model.generator, criterion, devices=devices, opt=model_opt)
        else:
            model.eval()
            val_loss_fn = train_utils.LossCompute(model.generator, criterion, model_opt)

        print("Validation...")
        loss, bleu_loss = train_utils.run_epoch(args, (train_utils.rebatch(pad_idx, b) for b in valid_iter), model_parallel,
                                        val_loss_fn, valid_params=valid_params, is_valid=True)

        if bleu_loss > best_bleu_loss:
            best_bleu_loss = bleu_loss

            model_state_dict = model.state_dict()
            model_file = args.save_to + args.exp_name + 'valid.bin'
            checkpoint = {
                'model': model_state_dict,
            }

            print('save model without optimizer [%s]' % model_file, file=sys.stderr)

            torch.save(checkpoint, model_file)

        print()
        print("Validation perplexity ", np.exp(loss))


def test(args):
    train_data, val_data, test_data, SRC, TGT = prepare_data(args)

    BATCH_SIZE = args.batch_size
    best_bleu_loss = 0
    pad_idx = TGT.vocab.stoi["<pad>"]
    print("Size of source vocabulary:", len(SRC.vocab))
    print("Size of target vocabulary:", len(TGT.vocab))

    print("FC matrix:", args.hidden_dim, args.ff_dim)
    model = transformer.make_model(len(SRC.vocab), len(TGT.vocab),
                                   d_model=args.hidden_dim, d_ff=args.ff_dim,\
                                   N=args.num_blocks, compress=args.compress, \
                                    num_compress_enc = args.num_enc_blocks_comp,
                                    num_compress_dec = args.num_dec_blocks_comp)
    model.to(device)
    if args.load_model:
        print('load model from [%s]' % args.load_model, file=sys.stderr)
        params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
        state_dict = params['model']
        # opts = params['']
        model.load_state_dict(state_dict)

    if args.debug:
        #fast check number of parameters
        model_full = transformer.make_model(len(SRC.vocab), len(TGT.vocab),
                                       d_model=args.hidden_dim, d_ff=args.ff_dim, \
                                       N=args.num_blocks, compress=False, \
                                       num_compress_enc=0,
                                       num_compress_dec=0)
        debug_compress_info(model_full,model)
        exit()

    criterion = train_utils.LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.to(device)

    if args.multi_gpu:
        devices = list(np.arange(args.num_devices))
        model_parallel = nn.DataParallel(model, device_ids=devices)

    test_iter = data.Iterator(test_data, batch_size=50, train=False, sort=False, repeat=False,
                                  device=device)
    print("Number of examples in test: ", len([_ for _ in test_iter]))

    # test_loss_fn = train_utils.LossCompute(model.generator, criterion, model_opt)

    os.makedirs(args.save_to_file, exist_ok=True)
    if args.multi_gpu:
        model_parallel.eval()

        bleu_loss = train_utils.test_decode(model_parallel.module, SRC, TGT, test_iter, 10000, \
                                to_words=True,
                               file_path=os.path.join(args.save_to_file, args.exp_name))
    else:
        model.eval()
        bleu_loss = train_utils.test_decode(model, SRC, TGT, test_iter, -1,\
                                            to_words=True,
                                            file_path=os.path.join(args.save_to_file, args.exp_name))
    print()
    # print("Test perplexity ", np.exp(loss))
    print("Total bleu:", bleu_loss)

if __name__ == '__main__':
    args = init_config()
    print(args, file=sys.stderr)
    # seed the RNG
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
