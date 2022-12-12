import json
import os
import configparser


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


def report_f1(path,
              selected_indices,
              unlabeled_path,
              logger):
    ...