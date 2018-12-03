import torch
from torchtext import data, datasets
import spacy

def tokenize_de(text):
    return [tok.text for tok in spacy_src.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_tgt.tokenizer(text)]

def prepare_data(args):
    
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    
    if args.tokenize:
        spacy_src = spacy.load('en')
        spacy_tgt = spacy.load('de')
        
        SRC = data.Field(pad_token=BLANK_WORD, lower=args.lower, tokenize=tokenize_de)
        TGT = data.Field(init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=BLANK_WORD, 
                        lower=args.lower, tokenize=tokenize_en)
    else:
        
        SRC = data.Field(pad_token=BLANK_WORD, lower=args.lower)
        TGT = data.Field(init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=BLANK_WORD, 
                        lower=args.lower)

    MAX_LEN = args.max_lenght

    train_data, val_data, test_data = datasets.IWSLT.splits(
        exts=('.de', '.en'), fields=(SRC, TGT),
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                              len(vars(x)['trg']) <= MAX_LEN
    )
    MIN_FREQ = args.min_freq
    SRC.build_vocab(train_data.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train_data.trg, min_freq=MIN_FREQ)
    return train_data, val_data, test_data, SRC, TGT
