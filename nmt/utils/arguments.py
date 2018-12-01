import argparse


def init_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--cuda', action='store_true', default=False, help='use gpu')

    parser.add_argument('--mode', choices=['train', 'test', 'compress',], default='train', help='run mode')
    parser.add_argument('--vocab', type=str, help='path to the vocabulary')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--beam_size', default=5, type=int, help='beam size for beam search')

    #  Model parameters

    # Data
    parser.add_argument('--dataset', type=str, help='type of the dataset')

    parser.add_argument('--train_src', type=str, help='path to the training source file')
    parser.add_argument('--train_tgt', type=str, help='path to the training target file')
    parser.add_argument('--dev_src', type=str, help='path to the dev source file')
    parser.add_argument('--dev_tgt', type=str, help='path to the dev target file')
    parser.add_argument('--test_src', type=str, help='path to the test source file')
    parser.add_argument('--test_tgt', type=str, help='path to the test target file')

    parser.add_argument('--decode_max_time_step', default=200, type=int, help='maximum number of time steps used '
                                                                            'in decoding and sampling')

    # Compress parameters

    # Training parameters
    parser.add_argument('--valid_niter', default=800, type=int, help='every n iterations to perform validation')
    parser.add_argument('--valid_metric', default='bleu', choices=['bleu', 'ppl', 'word_acc', 'sent_acc'],\
                        help='metric used for validation')
    parser.add_argument('--log_every', default=400, type=int, help='every n iterations to log training statistics')
    parser.add_argument('--load_model', default=None, type=str, help='path to the pre-trained model')
    parser.add_argument('--save_to', default='model', type=str, help='save trained model to')
    parser.add_argument('--save_model_after', default=0, type=int, help='save the model only after n validation iterations')
    parser.add_argument('--save_to_file', default=None, type=str, help='if provided, save decoding results to file')
    parser.add_argument('--save_nbest', default=False, action='store_true', help='save nbest decoding results')
    parser.add_argument('--patience', default=5, type=int, help='training patience')
    parser.add_argument('--uniform_init', default=None, type=float, help='uniform initialization for parameters')
    parser.add_argument('--clip_grad', default=5., type=float, help='clip gradients')
    parser.add_argument('--max_epoch', default=10, type=int, help='maximum number of training iterations')

    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr_warm', default=0.0001, type=float, help='learning rate for warmed model')
    parser.add_argument('--lr_decay', default=0.5, type=float, help='decay learning rate if the validation performance drops')

    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--smooth_bleu', action='store_true', default=False,
                        help='smooth sentence level BLEU score.')
    parser.add_argument('--sentence_bleu', action='store_true', default=False,
                        help='use sentence level for bleu calculation.')

    args = parser.parse_args()

    return args