#!/usr/bin/env python3

import argparse
import os
import random
import shutil
import time
from typing import Dict

import tensorflow as tf

from textgenrnn import textgenrnn
from textgenrnn.run_utils import rnn_generate, rnn_guess, get_auto_name, get_paths


def run_train(args: argparse.Namespace):
    if not args.name:
        raise RuntimeError('Model name is required')
    if not os.path.exists(args.file_path):
        raise RuntimeError(f'{args.file_path} not found')
    model_name = get_auto_name(**get_auto_name_args(args))
    paths = get_paths(**get_paths_args(args))
    log_path = os.path.join(args.log_path, model_name)

    # Check if weight exists
    if os.path.exists(paths['model_dir']):
        if not os.path.exists(paths['weights_path']):
            shutil.rmtree(paths['model_dir'])
            print(f"--- Folder for model {model_name} found but no weights file is present, start new model. ---")
        elif args.retrain:
            shutil.rmtree(paths['model_dir'])
            print(f"--- Retrain requested, model {model_name} deleted ---")
    new_model = not os.path.exists(paths['model_dir'])
    if new_model:
        # Delete logs if they exist
        if os.path.exists(log_path):
            shutil.rmtree(log_path)
        os.makedirs(log_path)
        os.makedirs(paths['model_dir'])
    else:
        print(f"--- Model {model_name} already exists, resuming training ---")

    rnn = textgenrnn(name=model_name, new_model=new_model, **paths)
    kwargs = vars(args).copy()
    kwargs.pop('name')
    kwargs.pop('file_path')
    kwargs['log_path'] = log_path
    rnn.train_from_largetext_file(args.file_path, new_model=new_model, **kwargs)


def run_generate(args: argparse.Namespace):
    if not args.name and not args.model_dir:
        raise RuntimeError('Model name or model directory is required')
    paths = get_paths(**get_paths_args(args))
    if not os.path.exists(paths['weights_path']):
        raise RuntimeError(f"Weights file {paths['weights_path']} not found")
    start_time = time.perf_counter()
    paths.pop('model_dir')
    text, words = rnn_generate(**paths, min_words=args.min_words, temperature=args.temperature)
    print(text)
    print(f"Generated {words} words in {time.perf_counter() - start_time:.2f}s.")


def run_guess(args: argparse.Namespace):
    start_time = time.perf_counter()
    if args.models:
        check_models = args.models
    else:
        check_models = random.sample(os.listdir(args.models_dir), args.pick)
    in_words = ' '.join(args.text)
    print(f"Guessing who said '{in_words}' with {', '.join(check_models)}")
    guess_dict = rnn_guess(models_dir=args.models_dir, check_models=check_models, in_str=in_words)
    for k, v in guess_dict.items():
        print(f'{k}: {v:.0f}%')
    print(f"Completed in {time.perf_counter() - start_time:.2f}s")


def get_auto_name_args(args: argparse.Namespace) -> Dict[str, str]:
    return dict(
        name=args.name,
        rnn_layers=args.rnn_layers,
        rnn_size=args.rnn_size,
        rnn_bidirectional=args.rnn_bidirectional,
        rnn_type=args.rnn_type,
    )


def get_paths_args(args: argparse.Namespace) -> Dict[str, str]:
    return dict(
        model_name=args.name,
        model_dir=args.model_dir,
        models_dir=args.models_dir,
        **get_auto_name_args(args),
    )


def main():
    parser = argparse.ArgumentParser(description='Runs the textgenrnn thing.')
    # Global options
    grp_global = parser.add_argument_group(title='Global options')
    grp_global.add_argument('-n', '--name', required=False, type=str,
                            help='Model name')
    grp_global.add_argument('--model-dir', required=False, type=str,
                            help='Specify model directory directly')
    grp_global.add_argument('--models-dir', required=False, type=str,
                            help='Path to models', default='./models')
    grp_global.add_argument('--type', dest='rnn_type', required=False, type=str,
                            help='Model type', default='lstm', choices=['lstm', 'gru'])
    grp_global.add_argument('--layers', dest='rnn_layers', required=False, type=int,
                            help='Number of layers', default=3)
    grp_global.add_argument('--size', dest='rnn_size', required=False, type=int,
                            help='RNN size', default=128)
    grp_global.add_argument('--bidir', dest='rnn_bidirectional', required=False, action='store_true',
                            help='Use Bidirectional RNN')
    grp_global.add_argument('--cpu', required=False, action='store_true',
                            help='Ignore GPUs')
    grp_global.add_argument('--allow-growth', required=False, action='store_true',
                            help='Allow GPU growth')
    grp_global.add_argument('--gpu-mem', required=False, type=int,
                            help='Limit GPU memory, applies to all GPUs', default=0)
    grp_global.add_argument('--verbose', required=False, type=int,
                            help='Verbosity level, 0 silent, 1 progress bar, 2 epoch only.', default=1)

    subparsers = parser.add_subparsers(title='Commands', required=True)

    # Training
    parser_train = subparsers.add_parser('train', help='Train model')
    parser_train.set_defaults(func=run_train)
    parser_train.add_argument('-i', '--file-path', required=True, type=str,
                              help='Path to text file', default='./')
    parser_train.add_argument('--log-path', required=False, type=str,
                              help='Path for TensorBoard logs', default='./logs')
    parser_train.add_argument('--retrain', required=False, action='store_true',
                              help='Delete existing model and start from scratch.')
    # Pass-through options
    parser_train.add_argument('--word-level', required=False, action='store_true',
                              help='Use word level')
    parser_train.add_argument('--train-size', required=False, type=float,
                              help='Train size fraction', default=0.95)
    parser_train.add_argument('--batch-size', required=False, type=int,
                              help='Specify batch size', default=2048)
    parser_train.add_argument('--dropout', required=False, type=float,
                              help='Dropout layer', default=0.0)
    parser_train.add_argument('--no-validation', dest='validation', required=False, action='store_false',
                              help='Disable validation')
    # Epoch stuff
    parser_train.add_argument('--gen-epoch', required=False, type=float,
                              help='How often to generate while training', default=10)
    parser_train.add_argument('--max-gen-length', required=False, type=float,
                              help='Max words to generate while training', default=50)
    parser_train.add_argument('--num-epochs', required=False, type=int,
                              help='Max epochs', default=200)
    parser_train.add_argument('--save-epochs', required=False, type=int,
                              help='How often to save', default=20)
    # Early stop due to loss
    parser_train.add_argument('--loss-min-delta', required=False, type=float,
                              help='Delta for loss early stopping', default=0.05)
    parser_train.add_argument('--loss-patience', required=False, type=int,
                              help='Loss patience for early stopping', default=5)
    parser_train.add_argument('--loss-restore', required=False, action='store_true',
                              help='Restore best weights if exiting early')
    parser_train.add_argument('--loss-stop', required=False, type=float,
                              help='Stop at minimum loss', default=0.8)
    # Early stop due to validation loss
    parser_train.add_argument('--val-min-delta', required=False, type=float,
                              help='Delta for validation loss early stopping', default=0.03)
    parser_train.add_argument('--val-patience', required=False, type=int,
                              help='Validation loss patience for early stopping', default=10)
    parser_train.add_argument('--val-restore', required=False, action='store_true',
                              help='Restore best weights if exiting early')
    # Reduce learning rate on plateau
    parser_train.add_argument('--plat-min-delta', required=False, type=float,
                              help='Delta for LR reduction', default=0.03)
    parser_train.add_argument('--plat-factor', required=False, type=float,
                              help='Factor for LR reduction', default=0.2)
    parser_train.add_argument('--plat-patience', required=False, type=int,
                              help='Patience for LR reduction', default=4)
    parser_train.add_argument('--plat-cooldown', required=False, type=int,
                              help='Cooldown for LR reduction', default=2)
    # Generation
    parser_gen = subparsers.add_parser('generate', help='Generate')
    parser_gen.set_defaults(func=run_generate)
    parser_gen.add_argument('-w', '--min-words', required=False, type=int,
                            help='Minimum number of words to generate', default=50)
    parser_gen.add_argument('-t', '--temperature', required=False, type=float,
                            help='Temperature', default=0.5)
    # Guess
    parser_guess = subparsers.add_parser('guess', help='Guess which model said something (badly)')
    parser_guess.set_defaults(func=run_guess)
    parser_guess.add_argument('text', nargs='+',
                              help='Guess text')
    guess_grp = parser_guess.add_mutually_exclusive_group()
    guess_grp.add_argument('-m', '--models', type=str, action='append',
                           help='Use these models, specify multiple times')
    guess_grp.add_argument('--pick', required=False, type=int,
                           help='Randomly pick from available models', default=3)

    # Hide debug info
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    args = parser.parse_args()
    if args.cpu:
        tf.config.experimental.set_visible_devices([], 'GPU')
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            if args.allow_growth:
                tf.config.experimental.set_memory_growth(gpu, True)
            elif args.gpu_mem:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=args.gpu_mem)]
                )

    args.func(args)


if __name__ == '__main__':
    main()
