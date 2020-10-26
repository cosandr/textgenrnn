#!/usr/bin/env python3

import argparse
import json
import os
import random
import shutil
import time
from typing import Dict

import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from textgenrnn import textgenrnn
from textgenrnn.model import textgenrnn_model


def run_train(args: argparse.Namespace):
    if not args.name:
        raise RuntimeError('Model name is required')
    if not os.path.exists(args.file_path):
        raise RuntimeError(f'{args.file_path} not found')
    model_name = get_auto_name(args)
    paths = get_paths(args, model_name=model_name)
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


def run_gen(args: argparse.Namespace):
    if not args.name and not args.model_dir:
        raise RuntimeError('Model name or model directory is required')
    model_name = get_auto_name(args)
    paths = get_paths(args, model_name=model_name)
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
    for name in check_models:
        paths = get_paths(args, name)
        # Load configs
        with open(paths['config_path'], 'r', encoding='utf8', errors='ignore') as json_file:
            config = json.load(json_file)
        with open(paths['vocab_path'], 'r', encoding='utf8', errors='ignore') as json_file:
            vocab = json.load(json_file)
        # Prepare vars
        num_classes = len(vocab) + 1
        # Build model
        start_time = time.perf_counter()
        model = textgenrnn_model(num_classes, cfg=config, weights_path=paths['weights_path'])
        print(F"Built {name} in {time.perf_counter() - start_time:.2f}s")
        # Config vars
        maxlen = config['max_length']
        if len(model.inputs) > 1:
            model = tf.keras.models.Model(inputs=model.inputs[0], outputs=model.outputs[1])

        encoded = np.array([vocab.get(x, 0) for x in in_words[:-1]])
        encoded_text = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=maxlen)
        preds = np.asarray(model.predict(encoded_text, batch_size=1, verbose=1)[0]).astype('float64')
        pred_next = preds[vocab.get(in_words[-1], 0)]
        print(f'{name}: {pred_next * 100:.0f}%\n')
    print(f"Completed in {time.perf_counter() - start_time:.2f}s")


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
grp_global.add_argument('--gpu-frac', required=False, type=float,
                        help='Fraction of GPU memory to use, 0 for all', default=0.8)
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
parser_gen.set_defaults(func=run_gen)
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


def rnn_generate(config_path, vocab_path, weights_path, min_words=50, temperature=0.5):
    # Load configs
    with open(config_path, 'r', encoding='utf8', errors='ignore') as json_file:
        config = json.load(json_file)
    with open(vocab_path, 'r', encoding='utf8', errors='ignore') as json_file:
        vocab = json.load(json_file)
    # Prepare vars
    num_classes = len(vocab) + 1
    indices_char = {v: k for k, v in vocab.items()}
    # Build model
    model = textgenrnn_model(num_classes, cfg=config, weights_path=weights_path)
    # Config vars
    maxlen = config['max_length']
    # Start with random letter
    ret_str = np.random.choice(list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    if len(model.inputs) > 1:
        model = tf.keras.models.Model(inputs=model.inputs[0], outputs=model.outputs[1])

    num_words = 0
    num_char = 0
    while num_char < (min_words + 50) * 6:
        encoded = np.array([vocab.get(x, 0) for x in ret_str])
        encoded_text = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=maxlen)

        preds = np.asarray(model.predict(encoded_text, batch_size=1)[0]).astype('float64')
        if temperature is None or temperature == 0.0:
            index = np.argmax(preds)
        else:
            preds = np.log(preds + tf.keras.backend.epsilon()) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            index = np.argmax(probas)
            # prevent function from being able to choose 0 (placeholder)
            # choose 2nd best index from preds
            if index == 0:
                index = np.argsort(preds)[-2]

        next_char = indices_char[index]
        ret_str += next_char
        num_char += 1
        if next_char == ' ':
            num_words += 1
        # Only stop after new line
        if (num_words >= min_words) and (next_char == '\n'):
            break

    return ret_str, num_words


def get_auto_name(args: argparse.Namespace):
    model_name = f"{args.name}_{args.rnn_layers}l{args.rnn_size}"
    if args.rnn_bidirectional:
        model_name += 'bi'
    if args.rnn_type == 'gru':
        model_name += '_gru'
    return model_name


def get_paths(args: argparse.Namespace, model_name='') -> Dict[str, str]:
    ret = {
        "model_dir": "",
        "config_path": "",
        "vocab_path": "",
        "weights_path": "",
    }
    # We have model directory, use that
    if args.model_dir:
        if not os.path.exists(args.model_dir):
            raise RuntimeError(f'{args.model_dir} does not exist')
        ret['model_dir'] = args.model_dir
        for f in os.listdir(args.model_dir):
            if f.endswith('_config.json'):
                ret['config_path'] = os.path.join(args.model_dir, f)
            elif f.endswith('_vocab.json'):
                ret['vocab_path'] = os.path.join(args.model_dir, f)
            elif f.endswith('_weights.hdf5') and '_epoch_' not in f:
                ret['weights_path'] = os.path.join(args.model_dir, f)
        # Check that we have all
        for k, v in ret.items():
            if not v:
                raise RuntimeError(f'Cannot find {k}')
    # Generate from model name and parameters
    else:
        if not model_name:
            model_name = get_auto_name(args)
        ret['model_dir'] = os.path.join(args.models_dir, model_name)
        ret['config_path'] = os.path.join(ret['model_dir'], f'{model_name}_config.json')
        ret['vocab_path'] = os.path.join(ret['model_dir'], f'{model_name}_vocab.json')
        ret['weights_path'] = os.path.join(ret['model_dir'], f'{model_name}_weights.hdf5')
    return ret


if __name__ == '__main__':
    # Hide debug info
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    _args = parser.parse_args()
    if _args.cpu:
        gpu_config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
        set_session(tf.compat.v1.Session(config=gpu_config))
    elif _args.gpu_frac:
        gpu_config = tf.compat.v1.ConfigProto()
        gpu_config.gpu_options.per_process_gpu_memory_fraction = _args.gpu_frac
        set_session(tf.compat.v1.Session(config=gpu_config))

    _args.func(_args)
