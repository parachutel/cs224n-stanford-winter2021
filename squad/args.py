"""Command-line arguments for setup.py, train.py, test.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--train_url',
                        type=str,
                        default='https://github.com/chrischute/squad/data/train-v2.0.json')
    parser.add_argument('--dev_url',
                        type=str,
                        default='https://github.com/chrischute/squad/data/dev-v2.0.json')
    parser.add_argument('--test_url',
                        type=str,
                        default='https://github.com/chrischute/squad/data/test-v2.0.json')
    parser.add_argument('--glove_url',
                        type=str,
                        default='http://nlp.stanford.edu/data/glove.840B.300d.zip')
    parser.add_argument('--dev_meta_file',
                        type=str,
                        default='./data/dev_meta.json')
    parser.add_argument('--test_meta_file',
                        type=str,
                        default='./data/test_meta.json')
    parser.add_argument('--word2idx_file',
                        type=str,
                        default='./data/word2idx.json')
    parser.add_argument('--char2idx_file',
                        type=str,
                        default='./data/char2idx.json')
    parser.add_argument('--answer_file',
                        type=str,
                        default='./data/answer.json')
    parser.add_argument('--include_test_examples',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Process examples from the test set')
    parser.add_argument('--train_record_file',
                        type=str,
                        default='./data/train.npz')
    parser.add_argument('--dev_record_file',
                        type=str,
                        default='./data/dev.npz')
    parser.add_argument('--test_record_file',
                        type=str,
                        default='./data/test.npz')
    parser.add_argument('--word_emb_file',
                        type=str,
                        default='./data/word_emb.json')
    parser.add_argument('--char_emb_file',
                        type=str,
                        default='./data/char_emb.json')
    parser.add_argument('--train_eval_file',
                        type=str,
                        default='./data/train_eval.json')
    parser.add_argument('--dev_eval_file',
                        type=str,
                        default='./data/dev_eval.json')
    parser.add_argument('--test_eval_file',
                        type=str,
                        default='./data/test_eval.json')
    
    # Data preprocessing and dimensions
    parser.add_argument('--glove_dim',
                        type=int,
                        default=300,
                        help='Size of GloVe word vectors to use')
    parser.add_argument('--glove_num_vecs',
                        type=int,
                        default=2196017,
                        help='Number of GloVe vectors')
    parser.add_argument('--para_limit',
                        type=int,
                        default=400,
                        help='Max number of words in a paragraph')
    parser.add_argument('--ques_limit',
                        type=int,
                        default=50,
                        help='Max number of words to keep from a question')
    parser.add_argument('--test_para_limit',
                        type=int,
                        default=1000,
                        help='Max number of words in a paragraph at test time')
    parser.add_argument('--test_ques_limit',
                        type=int,
                        default=100,
                        help='Max number of words in a question at test time')
    parser.add_argument('--char_dim',
                        type=int,
                        default=64,
                        help='Size of char vectors (char-level embeddings)')
    parser.add_argument('--ans_limit',
                        type=int,
                        default=30,
                        help='Max number of words in a training example answer')
    parser.add_argument('--char_limit',
                        type=int,
                        default=16,
                        help='Max number of chars to keep from a word')

    # Train
    parser.add_argument('--eval_steps',
                        type=int,
                        default=20000,
                        help='Number of steps between successive evaluations.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epochs for which to train. Negative means forever.')
    parser.add_argument('--metric_name',
                        type=str,
                        default='F1',
                        choices=('NLL', 'EM', 'F1'),
                        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=5,
                        help='Maximum number of checkpoints to keep on disk.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')
    parser.add_argument('--seed',
                        type=int,
                        default=223,
                        help='Random seed for reproducibility.')
    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.999,
                        help='Decay rate for exponential moving average of parameters.') 
    
    # Test
    parser.add_argument('--split',
                        type=str,
                        default='dev',
                        choices=('train', 'dev', 'test'),
                        help='Split to use for testing.')
    parser.add_argument('--sub_file',
                        type=str,
                        default='submission.csv',
                        help='Name for submission file.')
    parser.add_argument('--ensemble_mode',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Whether to use ensemble mode in test.')
    parser.add_argument('--run_analysis',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Whether to run analysis in test.')

        
    # Meta
    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=True,
                        help='Name to identify training or test run.')
    parser.add_argument('--max_ans_len',
                        type=int,
                        default=15,
                        help='Maximum length of a predicted answer.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of sub-processes to use per data loader.')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='Base directory for saving information.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=24,
                        help='Batch size per GPU. Scales automatically when \
                              multiple GPUs are available.')
    parser.add_argument('--use_squad_v2',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether to use SQuAD 2.0 (unanswerable) questions.')
    parser.add_argument('--num_visuals',
                        type=int,
                        default=10,
                        help='Number of examples to visualize in TensorBoard.')
    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint.')
    
    # Baseline
    parser.add_argument('--hidden_size',
                        type=int,
                        default=100,
                        help='Number of features in encoder hidden layers.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.5,
                        help='Learning rate for baseline.')
    parser.add_argument('--l2_wd',
                        type=float,
                        default=0,
                        help='L2 weight decay for baseline.')
    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.2,
                        help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--baseline_use_fusion',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether to use a linear fusion function after cq_att.')
    parser.add_argument('--baseline_use_char_emb',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether to use character embedding for baseline.')


    ############################################################################
    # QANet parameters #########################################################
    # See: #####################################################################
    # https://github.com/andy840314/QANet-pytorch-/blob/master/config.py
    # https://github.com/heliumsea/QANet-pytorch/blob/master/config.py
    ############################################################################
    parser.add_argument('--qanet_lr',
                        type=float,
                        default=0.001,
                        help='Learning rate.')
    parser.add_argument('--lr_warm_up_num',
                        type=int,
                        default=1000,
                        help='Number of steps for warming up training.')
    parser.add_argument('--n_encoder_blocks',
                        type=int,
                        default=7,
                        help='Number of layers of encoder blocks in QANet.')
    parser.add_argument('--n_head',
                        type=int,
                        default=8,
                        help='Number of attention heads in QANet.')
    parser.add_argument('--d_model',
                        type=int,
                        default=128,
                        help='Dimension of connectors in QANet.')
    parser.add_argument('--qanet_dropout',
                        type=float,
                        default=0.1,
                        help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--qanet_char_dropout',
                        type=float,
                        default=0.05,
                        help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--use_pretrained_char',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether to use pretrained character embeddings.')
    
    ############################################################################
    # QANet-XL parameters ######################################################
    ############################################################################
    parser.add_argument('--mem_len',
                        type=int,
                        default=128,
                        help='Length of memory segments')
    parser.add_argument('--d_head',
                        type=int,
                        default=16,
                        help='Dimension of att. head of RelPartialLearnableMultiHeadAttn.')

    args = parser.parse_args()
    if args.metric_name == 'NLL':
        # Best checkpoint is the one that minimizes negative log-likelihood
        args.maximize_metric = False
    elif args.metric_name in ('EM', 'F1'):
        # Best checkpoint is the one that maximizes EM or F1
        args.maximize_metric = True
    else:
        raise ValueError(f'Unrecognized metric name: "{args.metric_name}"')
    return args

args = get_args()