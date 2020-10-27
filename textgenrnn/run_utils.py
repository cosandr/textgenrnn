import json
import os
from typing import Dict, Tuple, List

import numpy as np
import tensorflow as tf
from keras import backend as K

from textgenrnn.model import textgenrnn_model


def rnn_generate(config_path: str, vocab_path: str, weights_path: str, min_words=50, temperature=0.5,
                 start_char='\n', reset=False) -> Tuple[str, int]:
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
    # Start with random letter
    if not start_char:
        ret_str = np.random.choice(list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    else:
        ret_str = start_char
    if len(model.inputs) > 1:
        model = tf.keras.models.Model(inputs=model.inputs[0], outputs=model.outputs[1])

    num_words = 0
    num_char = 0
    # Add 50 buffer, ~5 is the average length of an English word, add a bit more
    min_chars = (min_words + 50) * 6
    while num_char < min_chars:
        encoded = np.array([vocab.get(x, 0) for x in ret_str])
        encoded_text = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=config['max_length'])

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

    if reset:
        K.clear_session()
        tf.reset_default_graph()

    return ret_str, num_words


def rnn_guess(models_dir: str, check_models: List[str], in_str: str, reset=False) -> Dict[str, float]:
    """Keyword arguments for get_auto_name required"""
    ret_dict: Dict[str, float] = {}
    for name in check_models:
        paths = get_paths(models_dir=models_dir, model_name=name)
        # Load configs
        with open(paths['config_path'], 'r', encoding='utf8', errors='ignore') as json_file:
            config = json.load(json_file)
        with open(paths['vocab_path'], 'r', encoding='utf8', errors='ignore') as json_file:
            vocab = json.load(json_file)
        # Prepare vars
        num_classes = len(vocab) + 1
        # Build model
        model = textgenrnn_model(num_classes, cfg=config, weights_path=paths['weights_path'])
        # Config vars
        if len(model.inputs) > 1:
            model = tf.keras.models.Model(inputs=model.inputs[0], outputs=model.outputs[1])

        encoded = np.array([vocab.get(x, 0) for x in in_str[:-1]])
        encoded_text = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=config['max_length'])
        preds = np.asarray(model.predict(encoded_text, batch_size=1)[0]).astype('float64')
        pred_next = preds[vocab.get(in_str[-1], 0)]
        ret_dict[name] = pred_next * 100

    if reset:
        K.clear_session()
        tf.reset_default_graph()
    return ret_dict


def get_auto_name(name: str, rnn_layers: int, rnn_size: int, rnn_bidirectional=False, rnn_type='lstm') -> str:
    """Generate unique name from parameters"""
    model_name = f"{name}_{rnn_layers}l{rnn_size}"
    if rnn_bidirectional:
        model_name += 'bi'
    if rnn_type == 'gru':
        model_name += '_gru'
    return model_name


def get_paths(model_name='', model_dir='', models_dir='', **kwargs) -> Dict[str, str]:
    """Return model base directory and paths to config, vocab and weights

    If model_dir is specified, returns the respective files in it or raises RuntimeError.
    Otherwise, model_name, models_dir and parameters for get_auto_name are required.
    """
    ret = {
        "model_dir": "",
        "config_path": "",
        "vocab_path": "",
        "weights_path": "",
    }
    # We have model directory, use that
    if model_dir:
        if not os.path.exists(model_dir):
            raise RuntimeError(f'{model_dir} does not exist')
        ret['model_dir'] = model_dir
        for f in os.listdir(model_dir):
            if f.endswith('_config.json'):
                ret['config_path'] = os.path.join(model_dir, f)
            elif f.endswith('_vocab.json'):
                ret['vocab_path'] = os.path.join(model_dir, f)
            elif f.endswith('_weights.hdf5') and '_epoch_' not in f:
                ret['weights_path'] = os.path.join(model_dir, f)
        # Check that we have all
        for k, v in ret.items():
            if not v:
                raise RuntimeError(f'Cannot find {k}')
    # Generate from model name and parameters
    else:
        # Check args
        if not models_dir:
            raise TypeError('models_dir parameter is required')
        if not model_name:
            model_name = get_auto_name(**kwargs)
        ret['model_dir'] = os.path.join(models_dir, model_name)
        ret['config_path'] = os.path.join(ret['model_dir'], f'{model_name}_config.json')
        ret['vocab_path'] = os.path.join(ret['model_dir'], f'{model_name}_vocab.json')
        ret['weights_path'] = os.path.join(ret['model_dir'], f'{model_name}_weights.hdf5')
    return ret
