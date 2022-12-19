from methods.method_utils import *
import os
import subprocess
from methods.ker.auto_rules import *
from asp_solver.asp import *

DEFAULT_TRAIN_CONFIG_PATH = 'configs/conll04/example_train.conf'
TEMP_TRAIN_CONFIG_PATH = 'configs/conll04/example_train.conf.tmp'
DEFAULT_EVAL_CONFIG_PATH = 'configs/conll04/example_eval.conf'
TEMP_EVAL_CONFIG_PATH = 'configs/conll04/example_eval.conf.tmp'
DEFAULT_PREDICT_CONFIG_PATH = 'configs/conll04/example_predict.conf'
TEMP_PREDICT_CONFIG_PATH = 'configs/conll04/example_predict.conf.tmp'
TRAIN_SCRIPT = 'python ./spert.py train --config {config_path}'
EVAL_SCRIPT = 'python ./spert.py eval --config {config_path}'
PREDICT_SCRIPT = 'python ./spert.py predict --config {config_path}'


def ker(labeled_path,
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
        max_iter):

    # Create auto rules
    logger.info('Extracting auto rules')
    count = extract_auto_rules(unlabeled_path, auto_meta_path, True)
    logger.info(f'There are {count} sentences affected from auto rules')

    if not model_exists(os.path.join(labeled_model_path.format(-1), 'final_model')):
        os.makedirs(os.path.dirname(labeled_model_path.format(-1)), exist_ok=True)
        modify_config_file(DEFAULT_TRAIN_CONFIG_PATH,
                           TEMP_TRAIN_CONFIG_PATH,
                           {
                               'train_path': labeled_path,
                               'save_path': labeled_model_path.format(-1),
                               'log_path': train_log_path.format(-1)
                           })
        logger.info('Train on epoch -1')
        script = TRAIN_SCRIPT.format(config_path=TEMP_TRAIN_CONFIG_PATH)
        subprocess.run(script, shell=True, check=True)
    else:
        logger.info('Labeled model exists, skip training ...')

    logger.info('Evaluate on test data')
    modify_config_file(DEFAULT_EVAL_CONFIG_PATH,
                       TEMP_EVAL_CONFIG_PATH,
                       {
                           'model_path': os.path.join(labeled_model_path.format(-1), 'final_model'),
                           'log_path': eval_log_path.format(-1),
                           'tokenizer_path': os.path.join(labeled_model_path.format(-1), 'final_model')
                       })
    script = EVAL_SCRIPT.format(config_path=TEMP_EVAL_CONFIG_PATH)
    nmap_out = subprocess.run(script,
                              shell=True,
                              check=True,
                              universal_newlines=True,
                              stdout=subprocess.PIPE)
    nmap_lines = nmap_out.stdout.splitlines()
    filter_evaluation_log(nmap_lines, logger)

    iteration = 0
    while True:
        if iteration >= max_iter:
            break

        # Predict on unlabeled data
        modify_config_file(DEFAULT_PREDICT_CONFIG_PATH,
                           TEMP_PREDICT_CONFIG_PATH,
                           {
                               'model_path': os.path.join(labeled_model_path.format(iteration-1), 'final_model'),
                               'dataset_path': unlabeled_path,
                               'predictions_path': prediction_path
                           })
        script = PREDICT_SCRIPT.format(config_path=TEMP_PREDICT_CONFIG_PATH)
        logger.info(f'Round #{iteration}: Predict on unlabeled data')
        subprocess.run(script, shell=True, check=True)

        # Select labels using ASP
        logger.info(f'Round #{iteration}: Write prediction into files')
        write_pred_to_files(prediction_path, atom_meta_path)

        logger.info(f'Round #{iteration}: Solve using ASP')
        solve_all_docs(unlabeled_path=unlabeled_path,
                       atom_meta_path=atom_meta_path,
                       auto_meta_path=auto_meta_path,
                       selection_path=selection_path)

        # Unify labeled and selected pseudo labeled data
        logger.info(f'Round #{iteration}: Unify labels and pseudo labels')
        transfer_data(in_path1=labeled_path,
                      in_path2=prediction_path,
                      out_path=selection_path)

        # Compute F1 on selection
        logger.info(f'Round #{iteration}: F1 on selection')
        modify_config_file(DEFAULT_EVAL_CONFIG_PATH,
                           TEMP_EVAL_CONFIG_PATH,
                           {
                               'model_path': os.path.join(labeled_model_path.format(iteration-1), 'final_model'),
                               'log_path': eval_log_path.format(iteration + 0.5),
                               'tokenizer_path': os.path.join(labeled_model_path.format(iteration-1), 'final_model'),
                               'dataset_path': unlabeled_with_labels_path
                           })
        script = EVAL_SCRIPT.format(config_path=TEMP_EVAL_CONFIG_PATH)
        nmap_out = subprocess.run(script,
                                  shell=True,
                                  check=True,
                                  universal_newlines=True,
                                  stdout=subprocess.PIPE)
        nmap_lines = nmap_out.stdout.splitlines()
        filter_evaluation_log(nmap_lines, logger)

        # Create folder for models
        os.makedirs(os.path.dirname(labeled_model_path.format(iteration)), exist_ok=True)

        # Step 5: Retrain on labeled and pseudo-labeled data
        logger.info(f'Round #{iteration}: Retrain on selected pseudo labels')
        modify_config_file(DEFAULT_TRAIN_CONFIG_PATH,
                           TEMP_TRAIN_CONFIG_PATH,
                           {
                               'train_path': selection_path,
                               'save_path': labeled_model_path.format(iteration),
                               'log_path': train_log_path.format(iteration)
                           })
        script = TRAIN_SCRIPT.format(config_path=TEMP_TRAIN_CONFIG_PATH)
        subprocess.run(script, shell=True, check=True)

        # Step 6: Evaluate on test data
        logger.info(f'Round #{iteration}: Evaluate on test data')
        modify_config_file(DEFAULT_EVAL_CONFIG_PATH,
                           TEMP_EVAL_CONFIG_PATH,
                           {
                               'model_path': os.path.join(labeled_model_path.format(iteration), 'final_model'),
                               'log_path': eval_log_path.format(iteration),
                               'tokenizer_path': os.path.join(labeled_model_path.format(iteration), 'final_model')
                           })
        script = EVAL_SCRIPT.format(config_path=TEMP_EVAL_CONFIG_PATH)
        nmap_out = subprocess.run(script,
                                  shell=True,
                                  check=True,
                                  universal_newlines=True,
                                  stdout=subprocess.PIPE)
        nmap_lines = nmap_out.stdout.splitlines()
        filter_evaluation_log(nmap_lines, logger)
        iteration += 1




