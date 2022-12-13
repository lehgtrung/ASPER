from methods.curriculum_labeling.curriculum_labeling import curriculum_labeling
import os
import argparse
from logger import Logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run self training')
    parser.add_argument('--dataset',
                        action='store',
                        required=True)
    parser.add_argument('--fold',
                        type=int,
                        action='store',
                        required=True)
    parser.add_argument('--percent',
                        required=True,
                        type=int,
                        action='store')
    args = parser.parse_args()

    dataset = args.dataset
    fold = args.fold
    percent = args.percent
    method = 'curriculum_labeling'

    LABELED_PATH = f'./data/core_{dataset}/{dataset}_{percent}/fold_{fold}/labeled.json'
    UNLABELED_PATH = f'./data/core_{dataset}/{dataset}_{percent}/fold_{fold}/unlabeled.json'
    UNLABELED_WITH_LABELS_PATH = f'./data/core_{dataset}/{dataset}_{percent}/fold_{fold}/unlabeled_w_labels.json'
    PREDICTION_PATH = f'./data/methods/{method}/{dataset}_{percent}/fold_{fold}/prediction.json'
    SELECTION_PATH = f'./data/methods/{method}/{dataset}_{percent}/fold_{fold}/selection.json'
    LABELED_MODEL_PATH = './data/methods/{method}/{dataset}_{percent}/fold_{fold}/models/iter_{iter}/'
    LABELED_MODEL_PATH = LABELED_MODEL_PATH.format(
        method=method,
        dataset=dataset,
        percent=percent,
        fold=fold,
        iter='{}'
    )
    LOG_PATH = f'./data/methods/{method}/{dataset}_{percent}/fold_{fold}/logs.txt'
    TRAIN_LOG_PATH = './data/methods/{method}/{dataset}_{percent}/fold_{fold}/train_log/{iter}/'
    TRAIN_LOG_PATH = TRAIN_LOG_PATH.format(
        method=method,
        dataset=dataset,
        percent=percent,
        fold=fold,
        iter='{}'
    )
    EVAL_LOG_PATH = './data/methods/{method}/{dataset}_{percent}/fold_{fold}/eval_log/{iter}/'
    EVAL_LOG_PATH = EVAL_LOG_PATH.format(
        method=method,
        dataset=dataset,
        percent=percent,
        fold=fold,
        iter='{}'
    )

    os.makedirs(f'./data/methods/{method}/{dataset}_{percent}/fold_{fold}/models', exist_ok=True)

    logger = Logger(path=LOG_PATH)

    curriculum_labeling(labeled_path=LABELED_PATH,
                        unlabeled_path=UNLABELED_PATH,
                        train_log_path=TRAIN_LOG_PATH,
                        eval_log_path=EVAL_LOG_PATH,
                        prediction_path=PREDICTION_PATH,
                        selection_path=SELECTION_PATH,
                        labeled_model_path=LABELED_MODEL_PATH,
                        logger=logger)





