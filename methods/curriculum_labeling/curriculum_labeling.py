from methods.method_utils import *
import os
import subprocess
import time

DEFAULT_TRAIN_CONFIG_PATH = 'configs/{dataset}/example_train.conf'
TEMP_TRAIN_CONFIG_PATH = 'configs/{dataset}/example_train.conf.{hash_key}'
DEFAULT_EVAL_CONFIG_PATH = 'configs/{dataset}/example_eval.conf'
TEMP_EVAL_CONFIG_PATH = 'configs/{dataset}/example_eval.conf.{hash_key}'
DEFAULT_PREDICT_CONFIG_PATH = 'configs/{dataset}/example_predict.conf'
TEMP_PREDICT_CONFIG_PATH = 'configs/{dataset}/example_predict.conf.{hash_key}'
TRAIN_SCRIPT = 'python ./spert.py train --config {config_path}'
EVAL_SCRIPT = 'python ./spert.py eval --config {config_path}'
PREDICT_SCRIPT = 'python ./spert.py predict --config {config_path}'


def curriculum_labeling(dataset,
                        labeled_path,
                        unlabeled_path,
                        train_log_path,
                        eval_log_path,
                        prediction_path,
                        selection_path,
                        labeled_model_path,
                        logger,
                        delta=0.2):
    hash_key = '{0:010x}'.format(int(time.time() * 256))
    default_train_config_path = DEFAULT_TRAIN_CONFIG_PATH.format(dataset=dataset)
    default_eval_config_path = DEFAULT_EVAL_CONFIG_PATH.format(dataset=dataset)
    default_predict_config_path = DEFAULT_PREDICT_CONFIG_PATH.format(dataset=dataset)

    temp_train_config_path = TEMP_TRAIN_CONFIG_PATH.format(dataset=dataset, hash_key=hash_key)
    temp_eval_config_path = TEMP_EVAL_CONFIG_PATH.format(dataset=dataset, hash_key=hash_key)
    temp_predict_config_path = TEMP_PREDICT_CONFIG_PATH.format(dataset=dataset, hash_key=hash_key)

    if not model_exists(os.path.join(labeled_model_path.format(-1), 'final_model')):
        os.makedirs(os.path.dirname(labeled_model_path.format(-1)), exist_ok=True)
        modify_config_file(default_train_config_path,
                           temp_train_config_path,
                           {
                               'train_path': labeled_path,
                               'save_path': labeled_model_path.format(-1),
                               'log_path': train_log_path.format(-1)
                           })
        logger.info('Train on epoch -1')
        script = TRAIN_SCRIPT.format(config_path=temp_train_config_path)
        subprocess.run(script, shell=True, check=True)
    else:
        logger.info('Labeled model exists, skip training ...')

    logger.info('Evaluate on test data')
    modify_config_file(default_eval_config_path,
                       temp_eval_config_path,
                       {
                           'model_path': os.path.join(labeled_model_path.format(-1), 'final_model'),
                           'log_path': eval_log_path.format(-1),
                           'tokenizer_path': os.path.join(labeled_model_path.format(-1), 'final_model')
                       })
    script = EVAL_SCRIPT.format(config_path=temp_eval_config_path)
    nmap_out = subprocess.run(script,
                              shell=True,
                              check=True,
                              universal_newlines=True,
                              stdout=subprocess.PIPE)
    nmap_lines = nmap_out.stdout.splitlines()
    filter_evaluation_log(nmap_lines, logger)

    iteration = 0
    current_delta = 1.0 - delta
    while current_delta >= 0:
        logger.info(f'Round #{iteration}: Current delta: {current_delta}')
        # Predict on unlabeled data
        modify_config_file(default_predict_config_path,
                           temp_predict_config_path,
                           {
                               'model_path': os.path.join(labeled_model_path.format(iteration-1), 'final_model'),
                               'dataset_path': unlabeled_path,
                               'predictions_path': prediction_path
                           })
        script = PREDICT_SCRIPT.format(config_path=temp_predict_config_path)
        logger.info(f'Round #{iteration}: Predict on unlabeled data')
        subprocess.run(script, shell=True, check=True)

        # For each sentence, sort by minimum confidence
        logger.info(f'Round #{iteration}: Select pseudo labels by minimum probabilities')
        select_pseudo_labels_by_confidence(
            input_path=prediction_path,
            output_path=selection_path,
            z=current_delta
        )

        # Unify labeled and selected pseudo labeled data
        logger.info(f'Round #{iteration}: Unify labels and pseudo labels')
        transfer_data(in_path1=labeled_path,
                      in_path2=prediction_path,
                      out_path=selection_path)

        # Create folder for models
        os.makedirs(os.path.dirname(labeled_model_path.format(iteration)), exist_ok=True)

        # Retrain on labeled and pseudo-labeled data
        logger.info(f'Round #{iteration}: Retrain on selected pseudo labels')
        modify_config_file(default_train_config_path,
                           temp_train_config_path,
                           {
                               'train_path': selection_path,
                               'save_path': labeled_model_path.format(iteration),
                               'log_path': train_log_path.format(iteration)
                           })
        script = TRAIN_SCRIPT.format(config_path=temp_train_config_path)
        subprocess.run(script, shell=True, check=True)

        # Evaluate on test data
        logger.info(f'Round #{iteration}: Evaluate on test data')
        modify_config_file(default_eval_config_path,
                           temp_eval_config_path,
                           {
                               'model_path': os.path.join(labeled_model_path.format(iteration), 'final_model'),
                               'log_path': eval_log_path.format(iteration),
                               'tokenizer_path': os.path.join(labeled_model_path.format(iteration), 'final_model')
                           })
        script = EVAL_SCRIPT.format(config_path=temp_eval_config_path)
        nmap_out = subprocess.run(script,
                                  shell=True,
                                  check=True,
                                  universal_newlines=True,
                                  stdout=subprocess.PIPE)
        nmap_lines = nmap_out.stdout.splitlines()
        filter_evaluation_log(nmap_lines, logger)

        iteration += 1
        current_delta = current_delta - delta




