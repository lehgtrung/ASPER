#script (python)

import datetime
import string
import sys
import os
import operator

global curr_as
global max_as
global max_val
global debug
global listAll

def convert(clingoStr) :

     a = clingoStr[1:]
     a = a[:-1]

     return float(a)

def process_model(m):
    global curr_as
    global listAll
    global max_val
    global max_as

    curr_as = m.symbols(atoms=True)

    if (debug): print ("List of all elements: {}".format(curr_as), len(curr_as))

    pr = 1
    min_pr = 1.1
    avg_pr = 0
    no_atoms = 0
    relation_exist = False
    relation_flag = False
    for x in curr_as:
         if x.match("prod", 2) :
               valuePr = convert(str(x.arguments[0]))
               pr = pr * valuePr
               if min_pr > valuePr : min_pr = valuePr
               avg_pr = avg_pr+valuePr
               no_atoms = no_atoms + 1
         if x.match("invprod", 2) :
               pr = pr * (1.0 - convert(str(x.arguments[0])))
         if x.match("relation_exist",0):
               relation_exist = True
         if x.match("relation_flag",0):
               relation_flag = True

    # Trung, fix console problem
    #if pr <= 1e-7:
    #    pr = 0

    if no_atoms == 0:
        avg_pr = 0
        min_pr = 0
    else:
        avg_pr = avg_pr / no_atoms

    local_pr = min_pr
    if not relation_exist and relation_flag:
        pr = 0
        min_pr = 0
        avg_pr = 0

    max_as.append([[str(x) for x in m.symbols(atoms=True) if x.match("ok",1)], pr, local_pr, min_pr])

    #if pr > max_val:
         # max_val = pr
         # max_as.extend([[str(x) for x in m.symbols(atoms=True) if x.match("ok",1)], pr, min_pr, avg_pr])
         # max_as = [[str(x) for x in m.symbols(atoms=True) if x.match("ok",1)], pr, min_pr, avg_pr]
         # max_as = [[str(x) for x in m.symbols(atoms=True) if x.match("ok",1)], 0, min_pr, avg_pr]

    listAll.extend([[str(x) for x in m.symbols(atoms=True) if x.match("ok",1)], pr, min_pr, avg_pr])


def main(prg):

    global listAll
    global debug
    global max_val
    global max_as

    # debug = False
    debug = False
    max_val = -1
    listAll = []
    max_as = []

    # set option

    prg.configuration.solve.opt_mode = "optN"

    # compile the program

    prg.ground([("base",[])])

    prg.configuration.solve.models = 0

    prg.solve(on_model=process_model)

    #print("\n\n All optimal answer sets:",  listAll, len(listAll))

    #  listAll is the list of all answer sets with corresponding probability
    #  [ [list 1, prob of list 1], [ .. ], ...]
    #  max_as is the answer set with the maximal probability (max_val)

    if max_as:
        max_as = sorted(max_as, key=lambda x: (-x[1], -len(x[0]), -x[2]))[0]
        max_as = [max_as[0], 0, max_as[3]]

    print(max_as)
    # print(max_val)


    return

#end.


