import json
import random
import os
import sys
from tqdm import tqdm


def split_data_for_ssl(in_path, out_path, portion):
    with open(in_path, 'r') as f:
        data = json.load(f)
    n = len(data)
    m = int(n*portion)
    indices = list(range(n))
    random.shuffle(indices)
    indices = indices[:m]
    labeled = []
    unlabeled = []
    unlabeled_with_labels = []
    for i, row in enumerate(data):
        if i in indices:
            labeled.append(row)
        else:
            _row = {
                'tokens': row['tokens']
            }
            unlabeled.append(_row)
            unlabeled_with_labels.append(row)
    with open(out_path.format('labeled.json'), 'w') as f:
        json.dump(labeled, f)
    with open(out_path.format('unlabeled.json'), 'w') as f:
        json.dump(unlabeled, f)
    with open(out_path.format('unlabeled_w_labels.json'), 'w') as f:
        json.dump(unlabeled_with_labels, f)
    with open(out_path.format('indices.json'), 'w') as f:
        json.dump(indices, f)


def gen_data_folds(in_path, out_path, percent, num_folds):
    # Split the data into 30-70
    for i in tqdm(range(num_folds)):
        path = out_path.format(percent=percent, fold=i+1)
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, '{}')
        split_data_for_ssl(in_path, out_path=path, portion=percent/100)


if __name__ == '__main__':
    dataset = sys.argv[1]
    percent = int(sys.argv[2])
    num_folds = int(sys.argv[3])
    out_path = './data/core_{dataset}/{dataset}_{percent}/fold_{fold}'
    out_path = out_path.format(
        dataset=dataset,
        percent='{percent}',
        fold='{fold}'
    )
    gen_data_folds(in_path=f'./data/datasets/{dataset}/{dataset}_train.json',
                   out_path=out_path,
                   percent=percent,
                   num_folds=num_folds)