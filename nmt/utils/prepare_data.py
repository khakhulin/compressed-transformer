import os
import torch
from torchtext import data, datasets
from nltk.tokenize import ToktokTokenizer


class IWSLT14(datasets.TranslationDataset):
    """The IWSLT 2016 TED talk translation task"""

    name = 'iwslt14'
    base_dirname = '{}-{}'

    @classmethod
    def splits(cls, exts, fields, root='data',
               train='train', validation='valid',
               test='test', **kwargs):
        """Create dataset objects for splits of the IWSLT dataset.
        Arguments:
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            root: Root dataset storage directory. Default is '.data'.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """

        cls.dirname = cls.base_dirname.format(exts[0][1:], exts[1][1:])
        # cls.urls = [cls.base_url.format(exts[0][1:], exts[1][1:], cls.dirname)]
        # check = os.path.join(root, cls.name, cls.dirname)
        # path = cls.download(root, check=check)
        path = root
        print(cls.dirname)
        train = '.'.join([train, cls.dirname])
        validation = '.'.join([validation, cls.dirname])
        if test is not None:
            test = '.'.join([test, cls.dirname])

        if not os.path.exists(os.path.join(path, train) + exts[0]):
            cls.clean(path)

        train_data = None if train is None else cls(
            os.path.join(path, train), exts, fields, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), exts, fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), exts, fields, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


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
