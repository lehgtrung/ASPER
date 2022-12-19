import json
import re


def is_proper_name(s):
    if s in ['de', 'jr', 'O']:
        return True
    pattern = re.compile(r"^[A-Z][\D]+$")
    return bool(re.match(pattern, s))


def search_forward(tokens, idx, include):
    if include:
        start = idx
    else:
        start = idx + 1
    j = idx + 1
    while j < len(tokens):
        if is_proper_name(tokens[j]):
            j += 1
        else:
            break
    end = j
    return start, end


def search_backward(tokens, idx, include):
    if include:
        end = idx + 1
    else:
        end = idx
    j = idx - 1
    while j > 0:
        if is_proper_name(tokens[j]):
            j -= 1
        else:
            break
    start = j + 1
    return start, end


def match_people_with_title(tokens):
    titles = ['Mr.', 'Mrs.', 'Ms.', 'Miss.', 'President', 'Officer', 'Dr.', 'Professor', 'Sen.']
    result = []
    i = 0
    while i < len(tokens):
        end = i
        if tokens[i] in titles:
            start, end = search_forward(tokens, i, True)
            if end - start > 1:
                result.append(f'Peop("{start}+{end}").')
                # result.append(f'Peop("{tokens[start: end]}")')
        i = end + 1
    return result


def match_people_property(tokens):
    marker = ["'s"]
    i = 0
    result = []
    while i < len(tokens):
        end = i
        if tokens[i] in marker:
            start, end = search_backward(tokens, i, False)
            if end - start > 1:
                if tokens[i+1] == 'Party':
                    end = i + 2
                    result.append(f'Org("{start}+{end}").')
                    # result.append(f'Org("{tokens[start: end]}")')
                else:
                    # result.append(f'Prop_Owner("{tokens[start: end]}")')
                    result.append(f'Prop_Owner("{start}+{end}").')
        i = end + 1
    return result


def match_killing_of_people(tokens):
    i = 0
    result = []
    markers = ['murder', 'killer', 'killing', 'assassin', 'assassination', 'death']
    while i < len(tokens) - 2:
        end = i
        if tokens[i] in markers and tokens[i+1] == 'of':
            start, end = search_forward(tokens, i+1, False)
            if end - start > 1:
                result.append(f'dead("{start}+{end}").')
                # result.append(f'dead("{tokens[start: end]}")')
        i = end + 1
    return result


def match_people_killer(tokens):
    i = 0
    result = []
    markers = ['killer', 'murderer', 'assassin']
    while i < len(tokens) - 1:
        end = i
        if tokens[i] == "'s" and tokens[i+1] in markers:
            start, end = search_backward(tokens, i, True)
            if end - start > 1:
                result.append(f'dead("{start}+{end}").')
                # result.append(f'dead("{tokens[start: end]}")')
        i = end + 1
    return result


def match_company_name(tokens):
    i = 0
    result = []
    markers = ['Inc.', 'Co.']
    while i < len(tokens) - 1:
        end = i
        if tokens[i] in markers:
            start, end = search_backward(tokens, i, True)
            if end - start > 1:
                # result.append(f'Org("{tokens[start: end]}")')
                result.append(f'Org("{start}+{end}").')
        i = end + 1
    return result


def match_leader(tokens):
    i = 1
    result = []
    markers = ['leader', 'president', 'secretary', 'chairman',
               'director', 'governor', 'dean', 'head', 'chief']
    while i < len(tokens) - 2:
        end = i
        if (tokens[i] in markers and
                tokens[i + 1] == 'of' and
                tokens[i+2] == 'the' and
                tokens[i-1] == ','):
            start_f, end_f = search_forward(tokens, i+2, False)
            start_b, end_b = search_backward(tokens, i-1, False)
            if end_f - start_f > 1:
                if end_b - start_b > 1:
                    result.append(f'leader("{start_b}+{end_b}", "{start_f}+{end_f}").')
                    # result.append(f'leader("{tokens[start_b: end_b]}", "{tokens[start_f: end_f]}")')
                result.append(f'Org("{start_f}+{end_f}").')
                # result.append(f'Org("{tokens[start_f: end_f]}")')
        i = end + 1
    return result


def extract_auto_rules(input_path, output_path, empty):
    with open(input_path, 'r') as f:
        data = json.load(f)
    count = 0
    for i, row in enumerate(data):
        matches = (
            match_people_with_title(row['tokens']) +
            match_people_property(row['tokens']) +
            match_killing_of_people(row['tokens']) +
            match_people_killer(row['tokens']) +
            match_company_name(row['tokens']) +
            match_leader(row['tokens'])
        )
        matches = list(set(matches))
        if matches:
            count += 1
        if empty:
            open(output_path.format(i), 'w')
        else:
            with open(output_path.format(i), 'w') as f:
                f.writelines(map(lambda x: x + '\n', matches))
    return count


# if __name__ == '__main__':
    # with open('../../data/datasets/conll04/conll04_train.json', 'r') as f:
    #     data = json.load(f)
    # c = 0
    # for i, row in enumerate(data):
    #     matched = (
    #         match_people_with_title(row['tokens']) +
    #         match_people_property(row['tokens']) +
    #         match_killing_of_people(row['tokens']) +
    #         match_people_killer(row['tokens']) +
    #         match_company_name(row['tokens']) +
    #         match_leader(row['tokens'])
    #     )
    #     if matched:
    #         c += 1
    #         print(i, matched)
    # print(c)

    # input_path = '../../data/datasets/conll04/conll04_train.json'
    # extract_auto_rules(input_path, 'test/{}.txt')



