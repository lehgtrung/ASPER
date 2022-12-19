import os
import subprocess
import re
import json


COMMAND = 'clingo --opt-mode=optN asp_solver/p6.lp asp_solver/compute.lp {auto_path} {atom_path} ' \
          '--outf=0 -V0 --out-atomf=%s. --quiet=1,2,2'
entity_pattern = re.compile(r'(\w+)\("([0-9]+\+[0-9]+)"\)')
relation_pattern = re.compile(r'(\w+)\("([0-9]+\+[0-9]+)","([0-9]+\+[0-9]+)"\)')
ok_pattern = re.compile(r'ok\(\w+\).')


def match_form(atom, form):
    if form == 'entity':
        return bool(re.search(entity_pattern, atom))
    if form == 'relation':
        return bool(re.search(relation_pattern, atom))


def restore_entity(atom):
    match = re.findall(entity_pattern, atom)
    return {
        'type': match[0],
        'start': match[1].split('+')[0],
        'end': match[1].split('+')[1],
    }


def restore_relation(atom, entities):
    match = re.findall(entity_pattern, atom)
    head_ent = {
        'start': match[1].split('+')[0],
        'end': match[1].split('+')[1],
    }
    tail_ent = {
        'start': match[1].split('+')[0],
        'end': match[1].split('+')[1],
    }
    entities_wo_type = [{key: val for key, val in ent.items() if key != 'type'} for ent in entities]
    return {
        'type': match[0],
        'head': entities_wo_type.index(head_ent),
        'tail': entities_wo_type.index(tail_ent)
    }


def remove_ok(atom):
    return re.findall(ok_pattern, atom)[0]


def convert_atoms_to_doc(atoms, tokens, orig_id):
    entities = []
    relations = []
    for atom in atoms:
        atom = remove_ok(atom)
        if match_form(atom, 'entity'):
            entities.append(restore_entity(atom))
    for atom in atoms:
        if match_form(atom, 'relation'):
            relations.append(restore_relation(atom, entities))
    return {
        'tokens': tokens,
        'entities': entities,
        'relations': relations,
        'orig_id': orig_id
    }


def convert_to_atoms(pred):
    atoms = []
    entities = pred['entities']
    relations = pred['relations']
    for ent in entities:
        c = 'atom({}("{}"),"{}").'.format(ent['type'],
                                          '{}+{}'.format(ent['start'], ent['end']),
                                          ent['prob'])
        atoms.append(c)
    for rel in relations:
        head = entities[rel['head']]
        head_start, head_end = head['start'], head['end']
        head = f'{head_start}+{head_end}'
        tail = entities[rel['tail']]
        tail_start, tail_end = tail['start'], tail['end']
        tail = f'{tail_start}+{tail_end}'
        c = 'atom({}("{}","{}"),"{}").'.format(rel['type'],
                                               head,
                                               tail,
                                               rel['prob'])
        atoms.append(c)
    return atoms


def solve(command):
    # Write the program to a file
    process = subprocess.Popen(command,
                               stdin=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               stdout=subprocess.PIPE)
    output, error = process.communicate()
    result = [e.split() for e in output.decode().split('\n')[:-2]]
    return result


def solve_single_doc(unlabeled, auto_path, atom_path):
    command = COMMAND.format(
        auto_path=auto_path,
        atom_path=atom_path
    ).split()
    i = int(os.path.basename(auto_path).split('.')[0])
    j = int(os.path.basename(atom_path).split('.')[0])
    assert i == j
    result = solve(command)
    # Convert result to
    doc = convert_atoms_to_doc(atoms=result,
                               tokens=unlabeled[i]['tokens'],
                               orig_id=unlabeled[i]['orig_id'])
    return doc


def solve_all_docs(unlabeled_path, atom_meta_path, auto_meta_path, selection_path):
    with open(unlabeled_path, 'r') as f:
        unlabeled = json.load(f)
    new_pred = []
    for i, doc in enumerate(unlabeled):
        auto_path = auto_meta_path.format(i)
        atom_path = atom_meta_path.format(i)
        doc = solve_single_doc(unlabeled, auto_path, atom_path)
        new_pred.append(doc)
    with open(selection_path, 'w') as f:
        json.dump(new_pred, f)


# if __name__ == '__main__':
#     command = 'clingo --opt-mode=optN p6.lp compute.lp 2.txt --outf=0 -V0 --out-atomf=%s. --quiet=1,2,2'.split()
#
#     print(solve(command))


