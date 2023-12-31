import ast
import os
import subprocess
import re
import json
import numpy as np


COMMAND = 'clingo --opt-mode=optN asp_solver/{dataset}_solver.lp asp_solver/compute.lp {auto_path} {atom_path} ' \
          '--outf=0 -V0 --out-atomf=%s. --quiet=1,2,2'
entity_pattern = re.compile(r'^entity\((\w+),([0-9]+),([0-9]+)\)')
relation_pattern = re.compile(r'^relation\((\w+),([0-9]+),([0-9]+),([0-9]+),([0-9]+)\)')
ok_pattern = re.compile(r'^ok\((.*?)\)$')
syntactic_types = ['propOwner', 'dead']


def convert_doc_type_to_asp_type(dataset, atype, form):
    dct = {
        'conll04': {
            'entity': {
                'Peop': 'peop',
                'Other': 'other',
                'Loc': 'loc',
                'Org': 'org'
            },
            'relation': {
                'Work_For': 'workFor',
                'OrgBased_In': 'orgbasedIn',
                'Live_In': 'liveIn',
                'Located_In': 'locatedIn',
                'Kill': 'kill'
            }
        },
        'ade': {
            'entity': {
                'Adverse-Effect': 'adverseEffect',
                'Drug': 'drug'
            },
            'relation': {
                'Adverse-Effect': 'adverseEffect'
            }
        },
        'scierc': {
            'entity': {
                'Task': 'task',
                'Method': 'method',
                'Material': 'material',
                'OtherScientificTerm': 'otherScientificTerm',
                'Metric': 'metric',
                'Generic': 'generic'
            },
            'relation': {
                'Used-for': 'usedFor',
                'Feature-of': 'featureOf',
                'Hyponym-of': 'hyponymOf',
                'Evaluate-for': 'evaluateFor',
                'Part-of': 'partOf',
                'Compare': 'compare',
                'Conjunction': 'conjunction'
            }
        }
    }
    return dct[dataset][form][atype]


def convert_asp_type_to_doc_type(dataset, atype, form):
    dct = {
        'conll04': {
            'entity': {
                'peop': 'Peop',
                'other': 'Other',
                'loc': 'Loc',
                'org': 'Org'
            },
            'relation': {
                'workFor': 'Work_For',
                'orgbasedIn': 'OrgBased_In',
                'liveIn': 'Live_In',
                'locatedIn': 'Located_In',
                'kill': 'Kill'
            }
        },
        'ade': {
            'entity': {
                'adverseEffect': 'Adverse-Effect',
                'drug': 'Drug'
            },
            'relation': {
                'adverseEffect': 'Adverse-Effect'
            }
        },
        'scierc': {
            'entity': {
                'task': 'Task',
                'method': 'Method',
                'material': 'Material',
                'otherScientificTerm': 'OtherScientificTerm',
                'metric': 'Metric',
                'generic': 'Generic'
            },
            'relation': {
                'usedFor': 'Used-for',
                'featureOf': 'Feature-of',
                'hyponymOf': 'Hyponym-of',
                'evaluateFor': 'Evaluate-for',
                'partOf': 'Part-of',
                'compare': 'Compare',
                'conjunction': 'Conjunction'
            }
        }
    }
    return dct[dataset][form][atype]


def match_form(atom, form):
    if form == 'entity':
        return bool(re.search(entity_pattern, atom))
    if form == 'relation':
        return bool(re.search(relation_pattern, atom))


def restore_entity(dataset, atom):
    match = re.findall(entity_pattern, atom)
    match = match[0]
    if match[0] in syntactic_types:
        return None
    return {
        'type': convert_asp_type_to_doc_type(dataset, match[0], 'entity'),
        'start': int(match[1]),
        'end': int(match[2]),
    }


def restore_relation(dataset, atom, entities):
    match = re.findall(relation_pattern, atom)
    match = match[0]
    if match[0] in syntactic_types:
        return None
    head_ent = {
        'start': int(match[1]),
        'end': int(match[2]),
    }
    tail_ent = {
        'start': int(match[3]),
        'end': int(match[4]),
    }
    entities_wo_type = [{key: val for key, val in ent.items() if key != 'type'} for ent in entities]
    if head_ent in entities_wo_type and tail_ent in entities_wo_type:
        return {
            'type': convert_asp_type_to_doc_type(dataset, match[0], 'relation'),
            'head': entities_wo_type.index(head_ent),
            'tail': entities_wo_type.index(tail_ent)
        }
    return None


def remove_ok(atom):
    return re.findall(ok_pattern, atom)[0]


def convert_atoms_to_doc(dataset, atoms, prob, tokens):
    entities = []
    relations = []
    for atom in atoms:
        atom = remove_ok(atom)
        if match_form(atom, 'entity'):
            entity = restore_entity(dataset, atom)
            if entity:
                entities.append(entity)
    for atom in atoms:
        atom = remove_ok(atom)
        if match_form(atom, 'relation'):
            relation = restore_relation(dataset, atom, entities)
            if relation:
                relations.append(relation)
    return {
        'tokens': tokens,
        'entities': entities,
        'relations': relations,
        'prob': prob
    }


def convert_doc_to_atoms(dataset, pred):
    atoms = []
    entities = pred['entities']
    relations = pred['relations']
    for ent in entities:
        etype = convert_doc_type_to_asp_type(dataset, ent['type'], 'entity')
        start = ent['start']
        end = ent['end']
        prob = ent['prob']
        # c = f'atom({etype}({start},{end}),"{prob}").'
        c = f'atom(entity({etype},{start},{end}),"{prob}").'
        atoms.append(c)
    for rel in relations:
        rtype = convert_doc_type_to_asp_type(dataset, rel['type'], 'relation')
        head = entities[rel['head']]
        head_start, head_end = head['start'], head['end']
        tail = entities[rel['tail']]
        tail_start, tail_end = tail['start'], tail['end']
        prob = rel['prob']
        # c = f'atom({rtype}({head_start},{head_end},{tail_start},{tail_end}),"{prob}").'
        c = f'atom(relation({rtype},{head_start},{head_end},{tail_start},{tail_end}),"{prob}").'
        atoms.append(c)
    return atoms


def solve(command):
    # Write the program to a file
    process = subprocess.Popen(command,
                               stdin=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               stdout=subprocess.PIPE)
    output, error = process.communicate()
    # result = [e.split() for e in output.decode().split('\n')[:-2]]
    answerset = ast.literal_eval(output.decode().split('\n')[-2])
    if answerset:
        atoms = answerset[0]
        prob = answerset[2]
    else:
        atoms = []
        prob = 0
    return atoms, prob


def solve_single_doc(dataset, unlabeled, auto_path, atom_path):
    # print(atom_path)
    command = COMMAND.format(
        dataset=dataset,
        auto_path=auto_path,
        atom_path=atom_path
    ).split()
    i = int(os.path.basename(auto_path).split('.')[0])
    j = int(os.path.basename(atom_path).split('.')[0])
    assert i == j
    atoms, prob = solve(command)
    atoms = [e.replace(' ', '') for e in atoms]
    # Convert result to
    doc = convert_atoms_to_doc(dataset=dataset,
                               atoms=atoms,
                               prob=prob,
                               tokens=unlabeled[i]['tokens'])
    return doc, atoms


def solve_all_docs(dataset, unlabeled_path, atom_meta_path, auto_meta_path, selection_path):
    with open(unlabeled_path, 'r') as f:
        unlabeled = json.load(f)
    new_pred = []
    count_changes = 0
    for i, doc in enumerate(unlabeled):
        auto_path = auto_meta_path.format(i)
        atom_path = atom_meta_path.format(i)
        doc, atoms = solve_single_doc(dataset, unlabeled, auto_path, atom_path)
        if len(atoms) == 0:
            # raise ValueError(f'Empty atoms at #{i}')
            print(f'Empty atoms at #{i}')
        new_pred.append(doc)
        # if is_modified_by_asp(atom_path, ref_atoms=atoms):
        #     count_changes += 1
    with open(selection_path, 'w') as f:
        json.dump(new_pred, f)
    return count_changes


def solve_all_docs_with_curriculum(dataset, unlabeled_path, atom_meta_path,
                                   auto_meta_path, selection_path,
                                   with_curriculum, current_delta, logger):
    with open(unlabeled_path, 'r') as f:
        unlabeled = json.load(f)
    new_pred = []
    docs = []
    for i, doc in enumerate(unlabeled):
        auto_path = auto_meta_path.format(i)
        atom_path = atom_meta_path.format(i)
        doc, atoms = solve_single_doc(dataset, unlabeled, auto_path, atom_path)
        docs.append(doc)

    if with_curriculum:
        # threshold = np.percentile([d['prob'] for d in docs if d['prob'] > 0], current_delta * 100)
        threshold = np.percentile([d['prob'] for d in docs], current_delta * 100)
    else:
        threshold = 0
    for doc in docs:
        if doc['prob'] >= threshold:
            new_pred.append(doc)
        # if len(doc['relations']) > 0:
        #     new_pred.append(doc)
    with open(selection_path, 'w') as f:
        json.dump(new_pred, f)
    logger.info(f'Threshold: {threshold}')
    logger.info(f'Number of selected sentences: {len(new_pred)}')


def is_modified_by_asp(atom_path, ref_atoms):
    with open(atom_path, 'r') as f:
        lines = f.read()
    lines = [e for e in lines.split('\n') if e and e != '\n' and not e.startswith('%')]
    # convert every atom to ok and drop the probability
    oks = [e.replace('atom', 'ok') for e in lines]
    # oks = [remove_ok(e) for e in oks]
    atoms = []
    for atom in oks:
        if match_form(atom, 'entity'):
            match = re.findall(entity_pattern, atom)[0]
            atoms.append(f'ok({match[0]}({match[1]},{match[2]}))')
        else:
            match = re.findall(relation_pattern, atom)[0]
            atoms.append(f'ok({match[0]}({match[1]},{match[2]},{match[3]},{match[4]}))')
    return set(atoms) != set(ref_atoms)


if __name__ == '__main__':
    file_name = 'test_cases/conll04_test_texas.txt'
    command = f'clingo --opt-mode=optN conll04_solver.lp compute.lp {file_name} ' \
              f'--outf=0 -V0 --out-atomf=%s. --quiet=1,2,2'.split()

    print(solve(command))

    # atoms, prob = solve(command)
    # atoms = [e.replace(' ', '') for e in atoms]
    # # Convert result to
    # doc = convert_atoms_to_doc(atoms=atoms,
    #                            prob=prob,
    #                            tokens=[''])
    # print(doc)

    # ref_atoms = solve(command)
    # print(ref_atoms)
    # print(is_modified_by_asp(file_name, ref_atoms))


