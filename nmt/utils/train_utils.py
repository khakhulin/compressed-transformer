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
            self.ntokens = (self.trg_y != pad).sum().item()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & transformer.subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
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
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm
    

def evaluate_bleu(predictions, labels):
    try:
        bleu_nltk = nltk.translate.bleu_score.corpus_bleu(labels, predictions)
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
        src_mask = (src != SRC.vocab.stoi["<pad>"]).unsqueeze(-2)
        out = greedy_decode(model, src, src_mask,
                            max_len=50, start_symbol=TGT.vocab.stoi["<s>"])
        translate_str = []
        for j in range(1, out.size(1)):
            if to_words:
                sym = TGT.vocab.itos[out[0, j]]
                if sym == "</s>": break
            else:
                sym = out[0, j].item()
                if TGT.vocab.stoi["</s>"] == sym:
                    break
            translate_str.append(sym)
        tgt_str = []
        for j in range(1, batch.trg.size(0)):
            if to_words:
                sym = TGT.vocab.itos[batch.trg[j, 0]]
                if sym == "</s>": break
            else:
                sym = batch.trg[j, 0].item()
                if TGT.vocab.stoi["</s>"] == sym:
                    break
            tgt_str.append(sym)

        # else:
        #     translate_str = [str(i.item()) for i in out[0]]
        #     tgt_str = list(batch.trg[:, 0].cpu().numpy().astype(str))

        translate.append(translate_str)
        tgt.append([tgt_str])


        if (i + 1) % num_steps == 0:
            break
    print(translate[0])
    print(tgt[0][0])
    return evaluate_bleu(translate, tgt)

def run_epoch(args, data_iter, model, loss_compute, valid_params=None, epoch_num=0, is_valid=False):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    if valid_params is not None:
        src_dict, tgt_dict, valid_iter = valid_params
        
    bleu_all = 0
    count_all = 0

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
        if i % 100 == 1 and not is_valid:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f" %
                  (i, loss / float(batch.ntokens)))
            start = time.time()
            tokens = 0
           
        if (i + 1) % 500 == 0 and valid_params is not None and not is_valid:
            model.eval()
            bleu_val = valid(model, src_dict, tgt_dict, valid_iter, args.valid_max_num)
            print("BLEU ", bleu_val)
                

        if i % args.save_model_after == 0 and not is_valid:
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
            
    if is_valid:
        bleu_val = valid(model, src_dict, tgt_dict, valid_iter, 10000)
        print("BLEU (validation) ", bleu_val)

    return total_loss / total_tokens


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
        true_dist = torch.tensor(x, requires_grad=False)
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target == self.padding_idx)
        if len(mask) > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, torch.tensor(true_dist))

    
def rebatch(pad_idx, batch):
    "Fix order in torchtext"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           ys,
                           transformer.subsequent_mask(ys.size(1))
                                    .type_as(src))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.item()
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src).fill_(next_word)], dim=1)
    return ys