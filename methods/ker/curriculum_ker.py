from methods.method_utils import *
import os
import subprocess
import time
from methods.ker.auto_rules import *
from asp_solver.asp import *

DEFAULT_TRAIN_CONFIG_PATH = 'configs/{dataset}/example_train.conf'
TEMP_TRAIN_CONFIG_PATH = 'configs/{dataset}/example_train.conf.{hash_key}'
DEFAULT_EVAL_CONFIG_PATH = 'configs/{dataset}/example_eval.conf'
TEMP_EVAL_CONFIG_PATH = 'configs/{dataset}/example_eval.conf.{hash_key}'
DEFAULT_PREDICT_CONFIG_PATH = 'configs/{dataset}/example_predict.conf'
TEMP_PREDICT_CONFIG_PATH = 'configs/{dataset}/example_predict.conf.{hash_key}'
TRAIN_SCRIPT = 'python ./spert.py train --config {config_path}'
EVAL_SCRIPT = 'python ./spert.py eval --config {config_path}'
PREDICT_SCRIPT = 'python ./spert.py predict --config {config_path}'


def curriculum_ker(dataset,
                   labeled_path,
                   unlabeled_path,
                   unlabeled_with_labels_path,
                   train_log_path,
                   eval_log_path,
                   prediction_path,
                   atom_meta_path,
                   auto_meta_path,
                   selection_path,
                   labeled_model_path,
                   logger,
                   max_iter,
                   with_curriculum,
                   delta=0.2):
    hash_key = '{0:010x}'.format(int(time.time() * 256))
    default_train_config_path = DEFAULT_TRAIN_CONFIG_PATH.format(dataset=dataset)
    default_eval_config_path = DEFAULT_EVAL_CONFIG_PATH.format(dataset=dataset)
    default_predict_config_path = DEFAULT_PREDICT_CONFIG_PATH.format(dataset=dataset)

    temp_train_config_path = TEMP_TRAIN_CONFIG_PATH.format(dataset=dataset, hash_key=hash_key)
    temp_eval_config_path = TEMP_EVAL_CONFIG_PATH.format(dataset=dataset, hash_key=hash_key)
    temp_predict_config_path = TEMP_PREDICT_CONFIG_PATH.format(dataset=dataset, hash_key=hash_key)

    # Create auto rules
    logger.info('Extracting auto rules')
    count = extract_auto_rules(unlabeled_path, auto_meta_path, empty=True)
    logger.info(f'There are {count} sentences affected from auto rules')

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
    while iteration < max_iter:
        if current_delta < 0:
            break

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

        # Select labels using ASP
        logger.info(f'Round #{iteration}: Write prediction into files')
        write_pred_to_files(prediction_path, atom_meta_path)

        logger.info(f'Round #{iteration}: Solve using ASP')
        solve_all_docs_with_curriculum(unlabeled_path=unlabeled_path,
                                       atom_meta_path=atom_meta_path,
                                       auto_meta_path=auto_meta_path,
                                       selection_path=prediction_path + '.tmp',
                                       current_delta=current_delta,
                                       with_curriculum=with_curriculum,
                                       logger=logger)

        # Unify labeled and selected pseudo labeled data
        logger.info(f'Round #{iteration}: Unify labels and pseudo labels')
        transfer_data(in_path1=labeled_path,
                      in_path2=prediction_path + '.tmp',
                      out_path=selection_path)

        # Create folder for models
        os.makedirs(os.path.dirname(labeled_model_path.format(iteration)), exist_ok=True)

        # Step 5: Retrain on labeled and pseudo-labeled data
        logger.info(f'Round #{iteration}: Retrain on selected pseudo labels')
        # Fine tune instead of re-train, remove model_path if u dont want to
        modify_config_file(default_train_config_path,
                           temp_train_config_path,
                           {
                               'train_path': selection_path,
                               'save_path': labeled_model_path.format(iteration),
                               'log_path': train_log_path.format(iteration)
                           })
        script = TRAIN_SCRIPT.format(config_path=temp_train_config_path)
        subprocess.run(script, shell=True, check=True)

        # Step 6: Evaluate on test data
        logger.info(f'Round #{iteration}: Evaluate on test data')
        modify_config_file(default_eval_config_path,
                           temp_eval_config_path,
                           {
                               'model_path': os.path.join(labeled_model_path.format(iteration), 'final_model'),
                               'log_path': eval_log_path.format(iteration),
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




