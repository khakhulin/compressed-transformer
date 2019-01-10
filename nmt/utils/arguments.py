import argparse


def init_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--cuda', action='store_true', default=False, help='use gpu')

    parser.add_argument('--mode', choices=['train', 'test', 'compress',], default='train', help='run mode')
    parser.add_argument('--vocab', type=str, help='path to the vocabulary')

    parser.add_argument('--exp_name', default='simple', type=str, help='name of the experiment')

    #  Model parameters
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--hidden_dim', default=512, type=int, help='size of hidden dimention for all layers')
    parser.add_argument('--num_blocks', default=6, type=int, help='number of blocks')
    parser.add_argument('--ff_dim', default=2048, type=int, help='size of dimention for feed forward part')
    parser.add_argument('--num_enc_blocks_comp', default=6, type=int, help='number of compressed blocks')
    parser.add_argument('--num_dec_blocks_comp', default=6, type=int, help='number of compressed blocks')

    # Data
    parser.add_argument('--tokenize', action='store_true',  help='tokenize the dataset')
    parser.add_argument('--lower', action='store_true',  help='lowercase the dataset')
    parser.add_argument('--min_freq', type=int,  default=5, help='min frequency')
    parser.add_argument('--max_lenght', type=int,  default=50, help='max lenght of sentencess')

    parser.add_argument('--train_src', type=str, help='path to the training source file')
    parser.add_argument('--train_tgt', type=str, help='path to the training target file')
    parser.add_argument('--dev_src', type=str, help='path to the dev source file')
    parser.add_argument('--dev_tgt', type=str, help='path to the dev target file')
    parser.add_argument('--test_src', type=str, help='path to the test source file')
    parser.add_argument('--test_tgt', type=str, help='path to the test target file')

    parser.add_argument('--valid_max_num', default=100, type=int, help='maximum number of validation examples')
    parser.add_argument('--decode_max_time_step', default=200, type=int, help='maximum number of time steps used '
                                                                            'in decoding and sampling')

    parser.add_argument('--valid_every', default=150, type=int, help='how often validate bleu in epoch')

    # Compress parameters
    parser.add_argument('--compress', default=False, action='store_true', help='train compressed')
    parser.add_argument('--compress_mode', type=str, default='tt',
                        help='Decomposition for training in compressed mode: tt | tucker')
    parser.add_argument('--compress_attn', default=False, action='store_true', help='train with compressed matrix v')

    # Training parameters
    parser.add_argument('--valid_niter', default=800, type=int, help='every n iterations to perform validation')
    parser.add_argument('--valid_metric', default='bleu', choices=['bleu', 'ppl', 'word_acc', 'sent_acc'],\
                        help='metric used for validation')
    parser.add_argument('--log_every', default=400, type=int, help='every n iterations to log training statistics')
    parser.add_argument('--load_model', default=None, type=str, help='path to the pre-trained model')
    parser.add_argument('--save_to', default='saved_model/', type=str, help='save trained model to')
    parser.add_argument('--save_model_after', default=30, type=int, help='save the model only after n validation iterations')
    parser.add_argument('--save_to_file', default="files", type=str, help='if provided, save decoding results to file')
    parser.add_argument('--save_best', default=False, action='store_true', help='save best decoding results')
    parser.add_argument('--patience', default=5, type=int, help='training patience')
    parser.add_argument('--uniform_init', default=None, type=float, help='uniform initialization for parameters')
    parser.add_argument('--clip_grad', default=5., type=float, help='clip gradients')
    parser.add_argument('--max_epoch', default=10, type=int, help='maximum number of training iterations')

    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr_warm', default=0.0001, type=float, help='learning rate for warmed model')
    parser.add_argument('--lr_decay', default=0.5, type=float, help='decay learning rate if the validation performance drops')

    parser.add_argument('--debug', default=False, action='store_true')

    parser.add_argument('--multi-gpu', default=False, action='store_true')
    parser.add_argument('--num_devices', default=2, type=int, help='numbers of gpus')

    parser.add_argument('--smooth_bleu', action='store_true', default=False,
                        help='smooth sentence level BLEU score.')
    parser.add_argument('--sentence_bleu', action='store_true', default=False,
                        help='use sentence level for bleu calculation.')

    args = parser.parse_args()

    return args
