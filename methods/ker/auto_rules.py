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
        j = start + 1
    else:
        start = idx + 1
        j = start
    while j < len(tokens):
        if is_proper_name(tokens[j]):
            j += 1
        else:
            break
    end = j
    return start, end


def search_backward(tokens, idx):
    end = idx
    j = end - 1
    while j > 0:
        if is_proper_name(tokens[j]):
            j -= 1
        else:
            break
    start = j + 1
    return start, end


def match_people_with_title(tokens):
    titles = ['Mr.', 'Mrs.', 'Ms.', 'Miss.', 'President', 'Officer', 'Dr.', 'Professor', 'Sen.']
    stop = [',', '.']
    people = []
    i = 0
    while i < len(tokens):
        start = end = None
        if tokens[i] in titles:
            start = i
            j = i + 1
            while j < len(tokens):
                if tokens[j] not in stop and is_proper_name(tokens[j]):
                    j += 1
                else:
                    break
            end = i = j
        if start and end and (end - start > 1):
            # people.append(f'Peop("{start}+{end}", "1")')
            people.append(f'Peop("{tokens[start: end]}", "1")')
        i += 1
    return people


def match_people_property(tokens):
    marker = ["'s"]
    i = 0
    prop_owners = []
    while i < len(tokens):
        start = end = None
        if tokens[i] in marker:
            j = i - 1
            end = i
            while is_proper_name(tokens[j]) and j > 0:
                j -= 1
            start = j + 1
            i = end
        if (start and end) and (end - start > 1):
            if tokens[i+1] == 'Party':
                end = i + 2
                # prop_owners.append(f'Org("{start}+{end}", "1")')
                prop_owners.append(f'Org("{tokens[start: end]}", "1")')
            else:
                prop_owners.append(f'Prop_Owner("{tokens[start: end]}", "1")')
                # prop_owners.append(f'Prop_Owner("{start}+{end}", "1")')
        i += 1
    return prop_owners


def match_killing_of_people(tokens):
    i = 0
    people = []
    markers = ['murder', 'killer', 'killing', 'assassin', 'assassination', 'death']
    while i < len(tokens) - 1:
        start = end = None
        if tokens[i] in markers and tokens[i+1] == 'of':
            j = i + 2
            start = j
            while j < len(tokens):
                if is_proper_name(tokens[j]):
                    j += 1
                else:
                    break
            end = i = j
        if start and end and (end - start > 1):
            # people.append(f'Peop("{start}+{end}", "1")')
            people.append(f'Peop("{tokens[start: end]}", "1")')
        i += 1
    return people


def match_people_killer(tokens):
    i = 0
    people = []
    markers = ['killer', 'murderer', 'assassin']
    while i < len(tokens) - 1:
        start = end = None
        if tokens[i] == "'s" and tokens[i+1] in markers:
            j = i - 1
            end = i
            while is_proper_name(tokens[j]) and j > 0:
                j -= 1
            start = j + 1
            i = end
        if (start and end) and (end - start > 1):
            # people.append(f'Peop("{start}+{end}", "1")')
            people.append(f'Peop("{tokens[start: end]}", "1")')
        i += 1
    return people


def match_company_name(tokens):
    i = 0
    companies = []
    markers = ['Inc.', 'Co.']
    while i < len(tokens) - 1:
        start = end = None
        if tokens[i] in markers:
            j = i - 1
            end = i + 1
            while is_proper_name(tokens[j]) and j > 0:
                j -= 1
            start = j + 1
            i = end
        if (start and end) and (end - start > 1):
            companies.append(f'Org("{tokens[start: end]}", "1")')
            # companies.append(f'Org("{start}+{end}", "1")')
        i += 1
    return companies


def match_leader(tokens):
    i = 1
    people = []
    markers = ['leader', 'president', 'secretary', 'chairman',
               'director', 'governor', 'dean', 'head', 'chief']
    while i < len(tokens) - 2:
        start = end = None
        if (tokens[i] in markers and
                tokens[i + 1] == 'of' and
                tokens[i+2] == 'the' and
                tokens[i-1] == ','):
            j = i + 3
            start = j
            while j < len(tokens):
                if is_proper_name(tokens[j]):
                    j += 1
                else:
                    break
            end = i = j
        if start and end and (end - start > 1):
            # people.append(f'Peop("{start}+{end}", "1")')
            people.append(f'Org("{tokens[start: end]}", "1")')
        i += 1
    return people


if __name__ == '__main__':
    with open('../../data/datasets/conll04/conll04_train.json', 'r') as f:
        data = json.load(f)
    # print(data[900]['tokens'])
    # exit()
    c = 0
    for i, row in enumerate(data):
        # print(i, row['tokens'])
        matched = (
            match_people_with_title(row['tokens']) +
            match_people_property(row['tokens']) +
            match_killing_of_people(row['tokens']) +
            match_people_killer(row['tokens']) +
            match_company_name(row['tokens']) +
            match_leader(row['tokens'])
        )
        if matched:
            c += 1
            print(i, matched)
    print(c)



