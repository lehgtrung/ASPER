
from asp import *


def write_down_a_list(path, lst):
    with open(path, 'w') as f:
        f.writelines(map(lambda x: x + '\n', lst))


def test_overlap_rules():
    doc = {'tokens': ['Rogers', ',', 'a', 'native', 'of', 'Fort', 'Worth', ',', 'Texas',
                      ',', 'has', 'a', 'dry', 'wit', 'and', 'so', 'much', 'of',
                      'the', 'American', 'southwest', 'in', 'his', 'manner',
                      'and', 'voice', 'that', 'he', 'spends', 'a', 'lot', 'of',
                      'time', 'denying', 'he', "'s", 'related', 'to', 'the',
                      'famous', 'Oklahoma', 'cowboy', 'humorist', 'Will', 'Rogers', '.'],
           'entities': [{'type': 'Peop', 'start': 0, 'end': 1, 'prob': 0.9990965127944946},
                        {'type': 'Loc', 'start': 5, 'end': 7, 'prob': 0.9876447319984436},
                        {'type': 'Loc', 'start': 5, 'end': 9, 'prob': 0.9885520935058594},
                        {'type': 'Loc', 'start': 8, 'end': 9, 'prob': 0.9570575952529907},
                        {'type': 'Loc', 'start': 19, 'end': 20, 'prob': 0.9988160133361816},
                        {'type': 'Loc', 'start': 40, 'end': 41, 'prob': 0.9981266856193542},
                        {'type': 'Peop', 'start': 43, 'end': 45, 'prob': 0.9986903071403503}],
           'relations': [{'type': 'Live_In', 'head': 0, 'tail': 3, 'prob': 0.9983723759651184},
                         {'type': 'Live_In', 'head': 0, 'tail': 1, 'prob': 0.992712140083313},
                         {'type': 'Live_In', 'head': 0, 'tail': 2, 'prob': 0.9999511241912842},
                         {'type': 'Located_In', 'head': 1, 'tail': 3, 'prob': 0.9998809099197388},
                         {'type': 'Located_In', 'head': 2, 'tail': 3, 'prob': 0.9527393579483032},
                         {'type': 'Live_In', 'head': 6, 'tail': 3, 'prob': 0.9992928504943848},
                         {'type': 'Live_In', 'head': 6, 'tail': 5, 'prob': 0.993676483631134},
                         {'type': 'Live_In', 'head': 6, 'tail': 2, 'prob': 0.9873551726341248}],
           'orig_id': 5281}
    atoms = convert_doc_to_atoms(doc)
    atom_path = 'test_cases/test_overlap.lp'
    write_down_a_list(atom_path, atoms)

    command = f'clingo --opt-mode=optN p6_index.lp compute.lp {atom_path}' \
              f' --outf=0 -V0 --out-atomf=%s. --quiet=1,2,2'.split()
    result = solve(command)
    expectation = ['ok(peop(0,1))', 'ok(loc(5,7))', 'ok(loc(8,9))',
                   'ok(loc(19,20))', 'ok(loc(40,41))', 'ok(peop(43,45))',
                   'ok(liveIn(0,1,8,9))', 'ok(liveIn(0,1,5,7))',
                   'ok(locatedIn(5,7,8,9))', 'ok(liveIn(43,45,8,9))', 'ok(liveIn(43,45,40,41))']
    assert set(result) == set(expectation)


if __name__ == '__main__':
    test_overlap_rules()


