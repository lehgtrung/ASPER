import json
import sys


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)
    i = int(sys.argv[2])
    print(data[i])


