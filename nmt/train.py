import os
import sys

import spacy
import torch.nn as nn
from torchtext import data, datasets

import nmt.transformer as transformer
import nmt.utils.optimizer as opt
import nmt.utils.train_utils as train_utils
from nmt.utils.arguments import init_config
from nmt.utils.utils import *


def to_input_variable(sents, vocab):
    """
    return a tensor of shape (src_sent_len, batch_size)
    """
    word_ids = word2id(sents, vocab)
    sents_t = input_transpose(word_ids, vocab['<pad>'], 50) #TODO max seq len bug
    sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
    return sents_var


def init_training(args):

    if args.load_model:
        params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        saved_args = params['args']
        state_dict = params['state_dict']

        model = Model(saved_args, vocab)
        model.load_state_dict(state_dict)
    else:
        print(args.vocab)
        vocab = torch.load(args.vocab)
        model = Model(args, vocab)

    model.train()
    model.to(device)

    if args.uniform_init:
        print('uniformly initialize parameters [-%f, +%f]' % (args.uniform_init, args.uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-args.uniform_init, args.uniform_init)

    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0
    nll_loss = nn.NLLLoss(weight=vocab_mask, size_average=False).to(device)
    cross_entropy_loss = nn.CrossEntropyLoss(weight=vocab_mask, size_average=False).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    return vocab, model, optimizer, nll_loss, cross_entropy_loss


# TODO remove
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


def prepare_data(args, spacy_src, spacy_tgt):

    def tokenize_de(text):
        return [tok.text for tok in spacy_src.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_tgt.tokenizer(text)]

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                     eos_token=EOS_WORD, pad_token=BLANK_WORD)

    MAX_LEN = 50

    train_data, val_data, test_data = datasets.IWSLT.splits(
        exts=('.de', '.en'), fields=(SRC, TGT),
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                              len(vars(x)['trg']) <= MAX_LEN
    )
    MIN_FREQ = 10
    SRC.build_vocab(train_data.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train_data.trg, min_freq=MIN_FREQ)
    return train_data, val_data, test_data, SRC, TGT


def train(args):
    spacy_en = spacy.load('en')
    spacy_de = spacy.load('de')
    train_data, val_data, test_data, SRC, TGT = prepare_data(args, spacy_de, spacy_en)

    BATCH_SIZE = args.batch_size

    pad_idx = TGT.vocab.stoi["<blank>"]

    # TODO : add model parameters to config
    # TODO : add loading model
    print("Size of target vocabulary:", len(TGT.vocab))

    model = transformer.make_model(len(SRC.vocab), len(TGT.vocab), d_model=512, d_ff=2048, N=6)
    model.to(device)

    if args.load_model:
        print('load model from [%s]' % args.load_model, file=sys.stderr)
        params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
        # TODO args = params['args']
        state_dict = params['model']
        # opts = params['']
        model.load_state_dict(state_dict)


    criterion = train_utils.LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.to(device)
    train_iter = train_utils.WrapperIterator(train_data, batch_size=BATCH_SIZE, device=device,
                                             repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                                             batch_size_fn=train_utils.batch_size_fn, train=True)
    valid_iter = train_utils.WrapperIterator(val_data, batch_size=BATCH_SIZE, device=device,
                                             repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                                             batch_size_fn=train_utils.batch_size_fn, train=False)

    model_opt = opt.WrapperOpt(model.src_embed[0].d_model, 1, 2000,
                                     torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9))

    # train_time = begin_time = time.time()
    valid_params = (SRC, TGT, valid_iter)

    print("Number of examples in train: ", len([_ for _ in train_iter]))
    print("Number of examples in validation: ", len([_ for _ in valid_iter]))

    os.makedirs(os.path.dirname(args.save_to), exist_ok=True)



    for epoch in range(args.max_epoch):

        model.train()
        train_utils.run_epoch(args, (train_utils.rebatch(pad_idx, b) for b in train_iter),
                              model,
                              train_utils.LossCompute(model.generator, criterion, model_opt),
                              valid_params=valid_params,
                              epoch_num = epoch)

        model.eval()
        print("Validation loss")
        loss = train_utils.run_epoch((train_utils.rebatch(pad_idx, b) for b in valid_iter),
                                     model,
                                     train_utils.LossCompute(model.generator, criterion, model_opt), valid_params=valid_params)
        print(loss)


    #
    # train_data_src = read_corpus(args.train_src, source='src')
    # train_data_tgt = read_corpus(args.train_tgt, source='tgt')
    #
    # dev_data_src = read_corpus(args.dev_src, source='src')
    # dev_data_tgt = read_corpus(args.dev_tgt, source='tgt')
    #
    # train_data = list(zip(train_data_src, train_data_tgt))
    # dev_data = list(zip(dev_data_src, dev_data_tgt))
    #
    # vocab, model, optimizer, nll_loss, cross_entropy_loss = init_training(args)
    #
    # train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    # cum_examples = cum_batches = report_examples = epoch = valid_num = best_model_iter = 0
    # hist_valid_scores = []



if __name__ == '__main__':
    args = init_config()
    print(args, file=sys.stderr)

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # print("Device {} available".format(device))
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.mode == 'train':
        train(args)



