import json
import os
import configparser
import numpy as np
from collections import Counter
from asp_solver.asp import *


def transfer_data(in_path1, in_path2, out_path):
    with open(in_path1, 'r') as f:
        data1 = json.load(f)
    with open(in_path2, 'r') as f:
        data2 = json.load(f)
    with open(out_path, 'w') as f:
        json.dump(data1 + data2, f)


def transfer_and_collect(in_path1, in_path2, out_path, threshold, current_indices):
    with open(in_path1, 'r') as f:
        data1 = json.load(f)
    with open(in_path2, 'r') as f:
        data2 = json.load(f)
    idx = []
    for i, row in enumerate(data2):
        min_prob = 1.1
        for ent in row['entities']:
            if ent['prob'] < min_prob:
                min_prob = ent['prob']
        for rel in row['relations']:
            if rel['prob'] < min_prob:
                min_prob = rel['prob']
        if min_prob >= threshold and i not in current_indices:
            idx.append(i)
    selected = [data2[i] for i in range(len(data2)) if i in idx]

    with open(out_path, 'w') as f:
        json.dump(data1 + selected, f)

    return current_indices + idx


def model_exists(path):
    if os.path.exists(os.path.join(path, 'pytorch_model.bin')):
        return True
    return False


def filter_evaluation_log(lines, logger):
    flag = False
    for line in lines:
        if line.startswith('Evaluation'):
            flag = True
        if flag:
            logger.raw_info(line)


def modify_config_file(config_path, new_config_path, new_configs):
    config = configparser.ConfigParser()
    config.read(config_path)
    for key, value in new_configs.items():
        config['1'][key] = value
    with open(new_config_path, 'w') as configfile:
        config.write(configfile)


def check_size(path):
    with open(path, 'r') as f:
        return len(json.load(f))


def global_agreement_ratio(paths):
    datasets = []
    for path in paths:
        with open(path, 'r') as f:
            datasets.append(json.load(f))
    assert len(set(len(e) for e in datasets)) == 1
    dataset_size = len(datasets[0])
    agreement = 0
    for i in range(dataset_size):
        entities_set = []
        relations_set = []
        for j in range(len(paths)):
            entities = set([f"{e['type']}({e['start']},{e['end']})" for e in datasets[j][i]['entities']])
            relations = set([f"{e['type']}({e['head']},{e['tail']})" for e in datasets[j][i]['relations']])
            entities_set.append(entities)
            relations_set.append(relations)
        flag = True
        for k in range(1, len(entities_set)):
            if entities_set[k] != entities_set[k-1]:
                flag = False
                break
        for k in range(1, len(relations_set)):
            if relations_set[k] != relations_set[k-1]:
                flag = False
                break
        if flag:
            agreement += 1
    return agreement / dataset_size


def select_agreement(in_path1, in_path2,
                     out_path):
    with open(in_path1, 'r') as f:
        dataset1 = json.load(f)
    with open(in_path2, 'r') as f:
        dataset2 = json.load(f)

    agreements = []
    dataset_size = len(dataset1)
    agreement_indices = []
    for i in range(dataset_size):
        entities1 = set([f"{e['type']}({e['start']},{e['end']})" for e in dataset1[i]['entities']])
        relations1 = set([f"{e['type']}({e['head']},{e['tail']})" for e in dataset1[i]['relations']])
        entities2 = set([f"{e['type']}({e['start']},{e['end']})" for e in dataset2[i]['entities']])
        relations2 = set([f"{e['type']}({e['head']},{e['tail']})" for e in dataset2[i]['relations']])

        if entities1 == entities2 and relations1 == relations2:
            agreements.append(dataset1[i])
            agreement_indices.append(i)
    with open(out_path, 'w') as f:
        json.dump(agreements, f)


def select_pseudo_labels_by_confidence(input_path, output_path, z):
    with open(input_path, 'r') as f:
        data = json.load(f)
    min_probs = []
    for i, row in enumerate(data):
        probs = []
        for ent in row['entities']:
            probs.append(ent['prob'])
        for rel in row['relations']:
            probs.append(rel['prob'])
        if len(probs) > 0:
            min_prob = min(probs)
        else:
            min_prob = 0
        min_probs.append(min_prob)
    top_z = int(len(data) * (1 - z))
    indices = list(np.asarray(min_probs).argsort()[-top_z:])
    new_data = []
    for i, row in enumerate(data):
        if i in indices:
            new_data.append(row)
    with open(output_path, 'w') as f:
        json.dump(new_data, f)


def calc_symbol_freq(symbols, n, threshold=0.5):
    counter = Counter(map(tuple, symbols))
    final_symbols = []
    for symbol in symbols:
        if counter[tuple(symbol)]/n >= threshold:
            final_symbols.append(symbol)
    return final_symbols


def convert_to_tuple(doc, dct):
    if 'start' in dct:
        return (dct['start'],
                dct['end'],
                dct['type'])
    # if dct['type'] in ['Loc', 'Peop', 'Org', 'Other']:
    #     return (dct['start'],
    #             dct['end'],
    #             dct['type'])
    return (doc['entities'][dct['head']]['start'],
            doc['entities'][dct['head']]['end'],
            doc['entities'][dct['head']]['type'],
            doc['entities'][dct['tail']]['start'],
            doc['entities'][dct['tail']]['end'],
            doc['entities'][dct['tail']]['type'],
            dct['type'])


def collect_symbols(preds, i, field):
    collection = []
    for pred in preds:
        collection.extend([convert_to_tuple(pred[i], e) for e in pred[i][field]])
    return collection


def aggregate_on_symbols(model_paths):
    preds = []
    outputs = []
    n = len(model_paths)
    for path in model_paths:
        with open(path, 'r') as f:
            preds.append(json.load(f))
    for i in range(len(preds[0])):
        tokens = preds[0][i]['tokens']
        symbols = collect_symbols(preds, i, 'entities')
        entities = calc_symbol_freq(symbols, n)
        symbols = collect_symbols(preds, i, 'relations')
        relations = calc_symbol_freq(symbols, n)
        outputs.append({
            'tokens': tokens,
            'entities': [tuple(e) for e in entities],
            'relations': [tuple(r) for r in relations]
        })
    return outputs


def write_down_a_list(path, lst):
    with open(path, 'w') as f:
        f.writelines(map(lambda x: x + '\n', lst))


def write_pred_to_files(prediction_path, meta_path):
    with open(prediction_path, 'r') as f:
        pred = json.load(f)
    for i, doc in enumerate(pred):
        path = meta_path.format(i)
        atoms = convert_doc_to_atoms(doc)
        write_down_a_list(path, atoms)

