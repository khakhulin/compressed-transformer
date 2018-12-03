import torch
from torchtext import data, datasets
from nltk.tokenize import ToktokTokenizer


def prepare_data(args):
    
    UNK_TOKEN = "<unk>"
    PAD_TOKEN = "<pad>"    
    SOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    
    if args.tokenize:
        toktok = ToktokTokenizer()        
        SRC = data.Field(unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=None, eos_token=EOS_TOKEN, 
                         lower=args.lower, tokenize=toktok.tokenize)
        TGT = data.Field(unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=SOS_TOKEN, eos_token=EOS_TOKEN, 
                        lower=args.lower, tokenize=toktok.tokenize)
    else:
        
        SRC = data.Field(unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=None, eos_token=EOS_TOKEN, lower=args.lower)
        TGT = data.Field(unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=SOS_TOKEN, eos_token=EOS_TOKEN, 
                        lower=args.lower)

    MAX_LEN = args.max_lenght

    train_data, val_data, test_data = datasets.Multi30k.splits(
        exts=('.de', '.en'), fields=(SRC, TGT),
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                              len(vars(x)['trg']) <= MAX_LEN
    )
    MIN_FREQ = args.min_freq
    SRC.build_vocab(train_data.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train_data.trg, min_freq=MIN_FREQ)
    return train_data, val_data, test_data, SRC, TGT
