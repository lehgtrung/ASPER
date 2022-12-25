import json


def check_dot(path):
    with open(path, 'r') as f:
        data = json.load(f)
    count = 0
    total = 0
    for i, row in enumerate(data):
        tokens = row['tokens']
        for ent in row['entities']:
            total += 1
            if '.' in tokens[ent['start']: ent['end']]:
                print(i)
                print(tokens[ent['start']: ent['end']])
                count += 1
    print(count / total)


if __name__ == '__main__':
    check_dot('/home/thoang/workspace/spert/data/datasets/conll04/conll04_train_dev.json')

