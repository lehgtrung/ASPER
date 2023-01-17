import json
import random

if __name__ == '__main__':
    with open('data/datasets/ade/ade_full.json', 'r') as f:
        data = json.load(f)

    indices = list(range(len(data)))
    random.shuffle(indices)

    print('len(data): ', len(data))

    train = data[:int(len(data) * 0.65)]
    dev = data[int(len(data) * 0.65):int(len(data) * 0.8)]
    test = data[int(len(data) * 0.8):]

    print('len(train): ', len(train))
    print('len(dev): ', len(dev))
    print('len(test): ', len(test))
    print('total: ', len(train)+len(dev)+len(test))

    with open('data/datasets/ade/ade_train.json', 'w') as f:
        json.dump(train, f)

    with open('data/datasets/ade/ade_dev.json', 'w') as f:
        json.dump(dev, f)

    with open('data/datasets/ade/ade_test.json', 'w') as f:
        json.dump(test, f)

