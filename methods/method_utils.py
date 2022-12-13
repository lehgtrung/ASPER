import json
import os
import configparser
import copy
import numpy as np


def transfer_data(in_path1, in_path2, out_path):
    with open(in_path1, 'r') as f:
        data1 = json.load(f)
    with open(in_path2, 'r') as f:
        data2 = json.load(f)
    with open(out_path, 'w') as f:
        json.dump(data1 + data2, f)


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
        min_prob = min(probs)
        min_probs.append(min_prob)
    top_z = int(len(data) * (1 - z))
    indices = list(np.asarray(min_probs).argsort()[-top_z:])
    new_data = []
    for i, row in enumerate(data):
        if i in indices:
            new_data.append(row)
    with open(output_path, 'w') as f:
        json.dump(new_data, f)

