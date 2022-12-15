import os
import subprocess
import numpy as np
import json
from methods.method_utils import *

DEFAULT_TRAIN_CONFIG_PATH = 'configs/conll04/example_train.conf'
TEMP_TRAIN_CONFIG_PATH = 'configs/conll04/example_train.conf.tmp'
DEFAULT_EVAL_CONFIG_PATH = 'configs/conll04/example_eval.conf'
TEMP_EVAL_CONFIG_PATH = 'configs/conll04/example_eval.conf.tmp'
DEFAULT_PREDICT_CONFIG_PATH = 'configs/conll04/example_predict.conf'
TEMP_PREDICT_CONFIG_PATH = 'configs/conll04/example_predict.conf.tmp'
TRAIN_SCRIPT = 'python ./spert.py train --config {config_path}'
EVAL_SCRIPT = 'python ./spert.py eval --config {config_path}'
PREDICT_SCRIPT = 'python ./spert.py predict --config {config_path}'
REEVAL_SCRIPT = 'python reevaluator.py --gt_path {gt_path} --pred_path {pred_path}'
DEFAULT_TEST_PATH = 'data/datasets/conll04/conll04_test.json'


def add_suffix_to_path(path, suffix, split_by):
    dir_name = os.path.dirname(path)
    base_name = os.path.basename(path)
    if split_by == '':
        mod_base_name = f'{suffix}'
    else:
        parts = base_name.split(split_by)
        mod_base_name = f'{parts[0]}_{suffix}{split_by}{parts[1]}'
    return os.path.join(dir_name, mod_base_name)


def evaluate_tri_training(gt_path, pred_path1, pred_path2, pred_path3, logger):
    aggregated_pred = aggregate_on_symbols([pred_path1, pred_path2, pred_path3])
    tmp_path = pred_path1 + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(aggregated_pred, f)
    script = REEVAL_SCRIPT.format(gt_path=gt_path, pred_path=tmp_path)
    print(script)
    nmap_out = subprocess.run(script,
                              shell=True,
                              check=True,
                              universal_newlines=True,
                              stdout=subprocess.PIPE)
    nmap_lines = nmap_out.stdout.splitlines()
    filter_evaluation_log(nmap_lines, logger)


def tri_training(labeled_path,
                 unlabeled_path,
                 train_log_path,
                 prediction_path,
                 test_prediction_path,
                 agreement_path,
                 selection_path,
                 labeled_model_path,
                 logger,
                 start_iter,
                 max_iter):
    # Boostrap 3 models
    labeled_paths = []
    labeled_model_paths = []
    prediction_paths = []
    selection_paths = []
    train_log_paths = []
    agreement_paths = []
    test_prediction_paths = []
    for i in range(3):
        labeled_paths.append(add_suffix_to_path(labeled_path, suffix=i, split_by='.'))
        labeled_model_paths.append(add_suffix_to_path(labeled_model_path, suffix=i, split_by=''))
        prediction_paths.append(add_suffix_to_path(prediction_path, suffix=i, split_by='.'))
        test_prediction_paths.append(add_suffix_to_path(test_prediction_path, suffix=i, split_by='.'))
        selection_paths.append(add_suffix_to_path(selection_path, suffix=i, split_by='.'))
        train_log_paths.append(add_suffix_to_path(train_log_path, suffix=i, split_by=''))
        agreement_paths.append(add_suffix_to_path(agreement_path, suffix=i, split_by='.'))

    # Train 3 models
    if not model_exists(os.path.join(labeled_model_paths[0].format(-1), 'final_model')):
        for i in range(3):
            with open(labeled_path, 'r') as f:
                data = json.load(f)
                sample = np.random.choice(data, len(data)).tolist()
            with open(labeled_paths[i], 'w') as f:
                # first iteration
                logger.info(f'Boostrap #{i} size: {len(sample)}')
                json.dump(sample, f)

        for i in range(3):
            os.makedirs(labeled_model_paths[i].format(-1), exist_ok=True)
            modify_config_file(DEFAULT_TRAIN_CONFIG_PATH,
                               TEMP_TRAIN_CONFIG_PATH,
                               {
                                   'train_path': labeled_paths[i],
                                   'save_path': labeled_model_paths[i].format(-1),
                                   'log_path': train_log_paths[i].format(-1)
                               })
            script = TRAIN_SCRIPT.format(config_path=TEMP_TRAIN_CONFIG_PATH)
            subprocess.run(script, shell=True, check=True)
    else:
        logger.info('Labeled model exists, skip training ...')

    # Make prediction for each model on test data at round -1
    for i in range(3):
        modify_config_file(DEFAULT_PREDICT_CONFIG_PATH,
                           TEMP_PREDICT_CONFIG_PATH,
                           {
                               'model_path': os.path.join(labeled_model_paths[i]
                                                          .format(-1), 'final_model'),
                               'dataset_path': DEFAULT_TEST_PATH,
                               'predictions_path': test_prediction_paths[i]
                           })
        script = PREDICT_SCRIPT.format(config_path=TEMP_PREDICT_CONFIG_PATH)
        logger.info(f'Round #{-1}: Predict on test data on model {i}')
        subprocess.run(script, shell=True, check=True)
    evaluate_tri_training(DEFAULT_TEST_PATH,
                          test_prediction_paths[0],
                          test_prediction_paths[1],
                          test_prediction_paths[2],
                          logger)

    iteration = start_iter
    while True:
        os.makedirs(os.path.dirname(labeled_model_paths[0].format(iteration)), exist_ok=True)
        if iteration >= max_iter:
            break

        # Make prediction for each model
        for i in range(3):
            modify_config_file(DEFAULT_PREDICT_CONFIG_PATH,
                               TEMP_PREDICT_CONFIG_PATH,
                               {
                                   'model_path': os.path.join(labeled_model_paths[i]
                                                              .format(iteration - 1), 'final_model'),
                                   'dataset_path': unlabeled_path,
                                   'predictions_path': prediction_paths[i]
                               })
            script = PREDICT_SCRIPT.format(config_path=TEMP_PREDICT_CONFIG_PATH)
            logger.info(f'Round #{iteration}: Predict on unlabeled data')
            subprocess.run(script, shell=True, check=True)

        # Stop when predictions from differs under a small ratio
        agreement_ratio = global_agreement_ratio([p for p in prediction_paths])
        logger.info(f'Round #{iteration}: Global agreement between 3 models: {agreement_ratio}')
        if agreement_ratio >= 0.9:
            logger.info(f'Round #{iteration}: Reach global agreement between 3 models')
            break

        # Otherwise, find agreements between models
        for i in range(2):
            for j in range(i+1, 3):
                logger.info(f'Round #{iteration}: Select agreement between model {i} and {j}')
                select_agreement(
                    in_path1=prediction_paths[i],
                    in_path2=prediction_paths[j],
                    out_path=agreement_paths[sum(range(3))-(i+j)]
                )

        # Transfer
        for i in range(3):
            logger.info(f'Round #{iteration}: Transfer agreement to selection on model {i}')
            transfer_data(in_path1=labeled_path,
                          in_path2=agreement_paths[i],
                          out_path=selection_paths[i])

        for i in range(3):
            logger.info(f'Round #{iteration}: Retrain on model {i}')
            modify_config_file(DEFAULT_TRAIN_CONFIG_PATH,
                               TEMP_TRAIN_CONFIG_PATH,
                               {
                                   'train_path': selection_paths[i],
                                   'save_path': labeled_model_paths[i].format(iteration),
                                   'log_path': train_log_paths[i].format(iteration)
                               })
            script = TRAIN_SCRIPT.format(config_path=TEMP_TRAIN_CONFIG_PATH)
            subprocess.run(script, shell=True, check=True)

        # Evaluate

        # Make prediction for each model on test data
        for i in range(3):
            modify_config_file(DEFAULT_PREDICT_CONFIG_PATH,
                               TEMP_PREDICT_CONFIG_PATH,
                               {
                                   'model_path': os.path.join(labeled_model_paths[i]
                                                              .format(iteration), 'final_model'),
                                   'dataset_path': DEFAULT_TEST_PATH,
                                   'predictions_path': test_prediction_paths[i]
                               })
            script = PREDICT_SCRIPT.format(config_path=TEMP_PREDICT_CONFIG_PATH)
            logger.info(f'Round #{iteration}: Predict on test data on model {i}')
            subprocess.run(script, shell=True, check=True)
        evaluate_tri_training(DEFAULT_TEST_PATH,
                              test_prediction_paths[0],
                              test_prediction_paths[1],
                              test_prediction_paths[2],
                              logger)
        iteration += 1



