import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets
import time, sys
import nmt.transformer as transformer
import nltk


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & transformer.subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        tgt_mask = torch.tensor(tgt_mask, dtype=torch.float, requires_grad=True, device=tgt.device)
        return tgt_mask
    

class LossCompute:
    "Wrapper for simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm.item()
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm.item()
    

def evaluate_bleu(predictions, labels):
    try:
        bleu_nltk = nltk.translate.bleu_score.corpus_bleu(
            labels, predictions, smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method2)
        # compared against bleu nltk without smoothing: for BLEU around 0.3 difference in 1e-4
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        print("\nWARNING: Could not compute BLEU-score. Error:", str(e))
        bleu_nltk = 0

    return bleu_nltk
    
    
def valid(model, SRC, TGT, valid_iter, num_steps, to_words=False):
    
    translate = []
    tgt = []
    for i, batch in enumerate(valid_iter):
        src = batch.src.transpose(0, 1)[:1]
        src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
        out = greedy_decode(model, src, src_mask,
                            max_len=60, start_symbol=TGT.vocab.stoi["<s>"])

        # print(evaluate_bleu([[[str(i.item()) for i in out[0]]]], [list(batch.trg.data[:, 0].cpu())]))

        if to_words:
            translate_str = []
            for i in range(1, out.size(1)):
                sym = TGT.vocab.itos[out[0, i]]
                if sym == "</s>": break
                translate_str.append(sym)

            tgt_str = []
            for i in range(1, batch.trg.size(0)):
                sym = TGT.vocab.itos[batch.trg.data[i, 0]]
                if sym == "</s>": break
                tgt_str.append(sym)
        else:
            translate_str = [str(i.item()) for i in out[0]]
            tgt_str = list(batch.trg.data[:, 0].cpu().numpy())

        translate.append(translate_str)
        tgt.append([tgt_str])

        if i % num_steps == 0:
            break

    return evaluate_bleu(translate, tgt)

def run_epoch(args, data_iter, model, loss_compute, valid_params=None, epoch_num=0):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    if valid_params is not None:
        src_dict, tgt_dict, valid_iter = valid_params

    for i, batch in enumerate(data_iter):
        # 2 x 25 x 512
        model.train()
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)

        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        # print(batch.ntokens, time.time() - start, loss, tokens)
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f" %
                  (i, loss / float(batch.ntokens)))
            start = time.time()
            tokens = 0
        #batch size 2x25 ? max_len//2 ?
        # print(greedy_decode(model, batch.src, batch.src_mask, max_len=30, start_symbol=1))

        if i % 100 == 0 and valid_params is not None:
            model.eval()
            bleu_val = valid(model, src_dict, tgt_dict, valid_iter, args.valid_max_num)
            print(bleu_val)
            exit()

        if i % args.save_model_after == 0:
            model_state_dict = model.state_dict()
            model_file = args.save_to + 'model.iter{}.epoch{}.bin'.format(i, epoch_num)

            checkpoint = {
                'model': model_state_dict,
                'opts': loss_compute.opt,
                'epoch': epoch_num
            }

            print('save model to [%s]' % model_file, file=sys.stderr)

            torch.save(checkpoint,model_file)

            print("")

    return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if len(mask) > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, torch.tensor(true_dist))


class WrapperIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def rebatch(pad_idx, batch):
    "Fix order in torchtext"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           ys,
                           transformer.subsequent_mask(ys.size(1))
                                    .type_as(src.data))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.item()
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys