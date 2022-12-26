#script (python)

import datetime
import string
import sys
import clingo
import os

#from clingo.symbol import Function

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
    for x in curr_as:
         if x.match("prod", 1) :
               pr = pr * convert(str(x.arguments[0]))
         if x.match("invprod", 1) :
               pr = pr * (1.0 - convert(str(x.arguments[0])))

    if max_val < pr :
         max_val = pr
         max_as =  [str(x) for x in m.symbols(atoms=True) if x.match("ok",1)]
    listAll.extend([[str(x) for x in m.symbols(atoms=True) if x.match("ok",1)], pr])


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

    # print("\n\n All optimal answer sets:",  listAll, len(listAll))

    #  listAll is the list of all answer sets with corresponding probability
    #  [ [list 1, prob of list 1], [ .. ], ...]
    #  max_as is the answer set with the maximal probability (max_val)

    # print("\n\n Optimal answer set:",  max_as, max_val)
    print(max_as)
    print(max_val)

    return

#end.


