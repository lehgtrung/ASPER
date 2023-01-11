% projection

atom(A) :- atom(A,P).
prob(A,P) :- atom(A,P).

prod(P, A) :- atom(A, P), ok(A).
invprod(P, A) :- atom(A, P), nok(A).

% Type requirements -- if the body is satisfied then one of the atom in the body is incorrect

2{bad(locatedIn(P1,P2,Q1,Q2)); bad(org(P1,P2))}:- atom(locatedIn(P1,P2,Q1,Q2)), atom(org(P1,P2)).
2{bad(locatedIn(P1,P2,Q1,Q2)); bad(peop(P1,P2))} :- atom(locatedIn(P1,P2,Q1,Q2)), atom(peop(P1,P2)).
2{bad(locatedIn(P1,P2,Q1,Q2)); bad(other(P1,P2))} :- atom(locatedIn(P1,P2,Q1,Q2)), atom(other(P1,P2)).
2{bad(locatedIn(P1,P2,Q1,Q2)); bad(org(Q1,Q2))}:- atom(locatedIn(P1,P2,Q1,Q2)), atom(org(Q1,Q2)).
2{bad(locatedIn(P1,P2,Q1,Q2)); bad(peop(Q1,Q2))} :- atom(locatedIn(P1,P2,Q1,Q2)), atom(peop(Q1,Q2)).
2{bad(locatedIn(P1,P2,Q1,Q2)); bad(other(Q1,Q2))}:- atom(locatedIn(P1,P2,Q1,Q2)), atom(other(Q1,Q2)).

2{bad(orgbasedIn(P1,P2,Q1,Q2)); bad(loc(P1,P2))}:- atom(orgbasedIn(P1,P2,Q1,Q2)), atom(loc(P1,P2)).
2{bad(orgbasedIn(P1,P2,Q1,Q2)); bad(peop(P1,P2))} :- atom(orgbasedIn(P1,P2,Q1,Q2)), atom(peop(P1,P2)).
2{bad(orgbasedIn(P1,P2,Q1,Q2)); bad(other(P1,P2))}:- atom(orgbasedIn(P1,P2,Q1,Q2)), atom(other(P1,P2)).
2{bad(orgbasedIn(P1,P2,Q1,Q2)); bad(org(Q1,Q2))}:- atom(orgbasedIn(P1,P2,Q1,Q2)), atom(org(Q1,Q2)).
2{bad(orgbasedIn(P1,P2,Q1,Q2)); bad(peop(Q1,Q2))} :- atom(orgbasedIn(P1,P2,Q1,Q2)), atom(peop(Q1,Q2)).
2{bad(orgbasedIn(P1,P2,Q1,Q2)); bad(other(Q1,Q2))}:- atom(orgbasedIn(P1,P2,Q1,Q2)), atom(other(Q1,Q2)).

2{bad(liveIn(P1,P2,Q1,Q2));bad(loc(P1,P2))}:- atom(liveIn(P1,P2,Q1,Q2)), atom(loc(P1,P2)).
2{bad(liveIn(P1,P2,Q1,Q2)); bad(org(P1,P2))}:- atom(liveIn(P1,P2,Q1,Q2)), atom(org(P1,P2)).
2{bad(liveIn(P1,P2,Q1,Q2)); bad(other(P1,P2))}:- atom(liveIn(P1,P2,Q1,Q2)), atom(other(P1,P2)).
2{bad(liveIn(P1,P2,Q1,Q2)); bad(peop(Q1,Q2))} :- atom(liveIn(P1,P2,Q1,Q2)), atom(peop(Q1,Q2)).
2{bad(liveIn(P1,P2,Q1,Q2)); bad(org(Q1,Q2))}:- atom(liveIn(P1,P2,Q1,Q2)), atom(org(Q1,Q2)).
2{bad(liveIn(P1,P2,Q1,Q2)); bad(other(Q1,Q2))}:- atom(liveIn(P1,P2,Q1,Q2)), atom(other(Q1,Q2)).

2{bad(workFor(P1,P2,Q1,Q2));bad(loc(P1,P2))}:- atom(workFor(P1,P2,Q1,Q2)), atom(loc(P1,P2)).
2{bad(workFor(P1,P2,Q1,Q2)); bad(org(P1,P2))}:- atom(workFor(P1,P2,Q1,Q2)), atom(org(P1,P2)).
2{bad(workFor(P1,P2,Q1,Q2)); bad(other(P1,P2))}:- atom(workFor(P1,P2,Q1,Q2)), atom(other(P1,P2)).
2{bad(workFor(P1,P2,Q1,Q2)); bad(peop(Q1,Q2))} :- atom(workFor(P1,P2,Q1,Q2)), atom(peop(Q1,Q2)).
2{bad(workFor(P1,P2,Q1,Q2));bad(loc(Q1,Q2))}:- atom(workFor(P1,P2,Q1,Q2)), atom(loc(Q1,Q2)).
2{bad(workFor(P1,P2,Q1,Q2)); bad(other(Q1,Q2))}:- atom(workFor(P1,P2,Q1,Q2)), atom(other(Q1,Q2)).

2{bad(kill(P1,P2,Q1,Q2));bad(loc(P1,P2))}:- atom(kill(P1,P2,Q1,Q2)), atom(loc(P1,P2)).
2{bad(kill(P1,P2,Q1,Q2)); bad(org(P1,P2))}:- atom(kill(P1,P2,Q1,Q2)), atom(org(P1,P2)).
2{bad(kill(P1,P2,Q1,Q2)); bad(other(P1,P2))}:- atom(kill(P1,P2,Q1,Q2)), atom(other(P1,P2)).
2{bad(kill(P1,P2,Q1,Q2));bad(loc(Q1,Q2))}:- atom(kill(P1,P2,Q1,Q2)), atom(loc(Q1,Q2)).
2{bad(kill(P1,P2,Q1,Q2));bad(org(Q1,Q2))}:- atom(kill(P1,P2,Q1,Q2)), atom(org(Q1,Q2)).
2{bad(kill(P1,P2,Q1,Q2)); bad(other(Q1,Q2))}:- atom(kill(P1,P2,Q1,Q2)), atom(other(Q1,Q2)).


% Other requirements
bad(liveIn(X1,X2,X1,X2)):- atom(liveIn(X1,X2,X1,X2)).
bad(locatedIn(X1,X2,X1,X2)):- atom(locatedIn(X1,X2,X1,X2)).
bad(orgbasedIn(X1,X2,X1,X2)):- atom(orgbasedIn(X1,X2,X1,X2)).
bad(workFor(X1,X2,X1,X2)):- atom(workFor(X1,X2,X1,X2)).
bad(kill(X1,X2,X1,X2)):- atom(kill(X1,X2,X1,X2)).

2{bad(peop(X1,X2)); bad(loc(X1,X2))} :- atom(peop(X1,X2)), atom(loc(X1,X2)).
2{bad(peop(X1,X2)); bad(org(X1,X2))} :- atom(peop(X1,X2)), atom(org(X1,X2)).
2{bad(peop(X1,X2)); bad(other(X1,X2))}:- atom(peop(X1,X2)), atom(other(X1,X2)).
2{bad(loc(X1,X2)); bad(org(X1,X2))}:- atom(loc(X1,X2)), atom(org(X1,X2)).
2{bad(loc(X1,X2)); bad(other(X1,X2))}:- atom(loc(X1,X2)), atom(other(X1,X2)).
2{bad(org(X1,X2)); bad(other(X1,X2))}:- atom(org(X1,X2)), atom(other(X1,X2)).

% Type requirements --
% the body of a constraint should not be satisfied

:- ok(locatedIn(P1,P2,Q1,Q2)), ok(org(P1,P2)).
:- ok(locatedIn(P1,P2,Q1,Q2)), ok(peop(P1,P2)).
:- ok(locatedIn(P1,P2,Q1,Q2)), ok(other(P1,P2)).
:- ok(locatedIn(P1,P2,Q1,Q2)), ok(org(Q1,Q2)).
:- ok(locatedIn(P1,P2,Q1,Q2)), ok(peop(Q1,Q2)).
:- ok(locatedIn(P1,P2,Q1,Q2)), ok(other(Q1,Q2)).

:- ok(orgbasedIn(P1,P2,Q1,Q2)), ok(loc(P1,P2)).
:- ok(orgbasedIn(P1,P2,Q1,Q2)), ok(peop(P1,P2)).
:- ok(orgbasedIn(P1,P2,Q1,Q2)), ok(other(P1,P2)).
:- ok(orgbasedIn(P1,P2,Q1,Q2)), ok(org(Q1,Q2)).
:- ok(orgbasedIn(P1,P2,Q1,Q2)), ok(peop(Q1,Q2)).
:- ok(orgbasedIn(P1,P2,Q1,Q2)), ok(other(Q1,Q2)).

:- ok(liveIn(P1,P2,Q1,Q2)), ok(loc(P1,P2)).
:- ok(liveIn(P1,P2,Q1,Q2)), ok(org(P1,P2)).
:- ok(liveIn(P1,P2,Q1,Q2)), ok(other(P1,P2)).
:- ok(liveIn(P1,P2,Q1,Q2)), ok(peop(Q1,Q2)).
:- ok(liveIn(P1,P2,Q1,Q2)), ok(org(Q1,Q2)).
:- ok(liveIn(P1,P2,Q1,Q2)), ok(other(Q1,Q2)).

:- ok(workFor(P1,P2,Q1,Q2)), ok(loc(P1,P2)).
:- ok(workFor(P1,P2,Q1,Q2)), ok(org(P1,P2)).
:- ok(workFor(P1,P2,Q1,Q2)), ok(other(P1,P2)).
:- ok(workFor(P1,P2,Q1,Q2)), ok(peop(Q1,Q2)).
:- ok(workFor(P1,P2,Q1,Q2)), ok(loc(Q1,Q2)).
:- ok(workFor(P1,P2,Q1,Q2)), ok(other(Q1,Q2)).

:- ok(kill(P1,P2,Q1,Q2)), ok(loc(P1,P2)).
:- ok(kill(P1,P2,Q1,Q2)), ok(org(P1,P2)).
:- ok(kill(P1,P2,Q1,Q2)), ok(other(P1,P2)).
:- ok(kill(P1,P2,Q1,Q2)), ok(loc(Q1,Q2)).
:- ok(kill(P1,P2,Q1,Q2)), ok(org(Q1,Q2)).
:- ok(kill(P1,P2,Q1,Q2)), ok(other(Q1,Q2)).

%% Other requirements

:- ok(liveIn(X1,X2,X1,X2)).
:- ok(locatedIn(X1,X2,X1,X2)).
:- ok(orgbasedIn(X1,X2,X1,X2)).
:- ok(workFor(X1,X2,X1,X2)).
:- ok(kill(X1,X2,X1,X2)).

:- ok(peop(X1,X2)), ok(loc(X1,X2)).
:- ok(peop(X1,X2)), ok(org(X1,X2)).
:- ok(peop(X1,X2)), ok(other(X1,X2)).
:- ok(loc(X1,X2)), ok(org(X1,X2)).
:- ok(loc(X1,X2)), ok(other(X1,X2)).
:- ok(org(X1,X2)), ok(other(X1,X2)).



%% Determine whether set of atom is inconsistent

inconsistency :- atom(X), bad(X).

%% Uncommented the line below if we do not want to use the output
%% whenever model 2 is inconsistent.

%% :- inconsistency(2).

%%% an atom should be kept if it does not violate any type constraint
%%% e.g., peop(x) is okay if there is no other atom indicating that it is a location, an organization, or other entity
%%%                      and x is not in any relation such as workfor(P,x) ...

ok(X)  :- atom(X), not bad(X).

%%% an atom could be kept if it contributes to the violation of some constraint
%%% or it is an inferred atom

{ ok(X) } :- atom(X), bad(X).

{ ok(X) } :- inf_atom(X), bad(X).

%%% when an atom is not kept then it is not okay

nok(X) :- atom(X), not ok(X).
nok(X) :- inf_atom(X), not ok(X).

%%% computing set difference

inSetDifference(X) :- ok(X), not atom(X).
inSetDifference(X) :- atom(X), nok(X).

%%% keeping the maximal number of atoms that can be kept

nOfAtoms(N) :- N = #count{1,X : inSetDifference(X)}.

%#minimize {N : nOfAtoms(N)}.

#show ok/1.
%% #show nOfAtoms/1.
%% #show bad/1.
%% #show atom/1.
%% #show dependency/2.
%% #show rule/3.

%%%%%%%%%%% inference rules %%%%%%%%%%%%%%%%%%%%

rule(locatedIn(P1,P2,Q1,Q2), orgbasedIn(O1,O2,P1,P2), orgbasedIn(O1,O2,Q1,Q2)):-
	atom(locatedIn(P1,P2,Q1,Q2)), atom(orgbasedIn(O1,O2,P1,P2)), P1 != P2, P1 != Q1.

rule(locatedIn(P1,P2,Q1,Q2), locatedIn(Q1,Q2,R1,r2), locatedIn(P1,P2,R1,r2)):-
	atom(locatedIn(P1,P2,Q1,Q2)), atom(locatedIn(Q1,Q2,R1,r2)), P1 != P2, P1 != Q1, Q1 != Q2, Q1 != R1.

rule(liveIn(X1,X2,P1,P2), locatedIn(P1,P2,Q1,Q2), liveIn(X1,X2,Q1,Q2)):-
	atom(liveIn(X1,X2,P1,P2)), atom(locatedIn(P1,P2,Q1,Q2)), P1 != P2, P1 != Q1.



%%%%% ASP for working with inference rules %%%%%%%%%%%%%%%%

6 {bad(X); bad(Y); bad(Z); inf_atom(Z);
    dependency(X, Z); dependency(Y, Z) } :-
       	rule(X, Y, Z), atom(X), atom(Y), not atom(Z).

:- ok(Y), inf_atom(Y), dependency(X,Y), not ok(X).
:- rule(X, Y, Z), ok(X), ok(Y), not ok(Z).



%% Trung: infer types for relations after they are selected
%% locatedIn/orgbasedIn/liveIn/workFor/kill
ok(loc(P1,P2)) :- ok(locatedIn(P1,P2,Q1,Q2)).
ok(loc(Q1,Q2)) :- ok(locatedIn(P1,P2,Q1,Q2)).
ok(org(O1,O2)) :- ok(orgbasedIn(O1,O2,P1,P2)).
ok(loc(P1,P2)) :- ok(orgbasedIn(O1,O2,P1,P2)).
ok(peop(X1,X2)) :- ok(liveIn(X1,X2,P1,P2)).
ok(loc(P1,P2)) :- ok(liveIn(X1,X2,P1,P2)).
ok(peop(X1,X2)) :- ok(workFor(X1,X2,O1,O2)).
ok(org(O1,O2)) :- ok(workFor(X1,X2,O1,O2)).
ok(peop(X1,X2)) :- ok(kill(X1,X2,Y1,Y2)).
ok(peop(Y1,Y2)) :- ok(kill(X1,X2,Y1,Y2)).


%% More rules
1{atom(peop(X1,X2),"1"); atom(org(X1,X2),"1"); atom(loc(X1,X2),"1")}1 :- atom(propOwner(X1,X2),"1").

%% Dead people can not kill
atom(peop(X1,X2),"1") :- atom(dead(X1,X2),"1").
:- atom(dead(X1,X2)), atom(kill(X1,X2,_)).

%% Reason about location's size
smaller(L1,L2,K1,K2) :- atom(locatedIn(L1,L2,K1,K2)).
smaller(L1,L2,J1,J2) :- smaller(L1,L2,K1,K2), smaller(K1,K2,J1,J2).
:- atom(locatedIn(K1,K2,L1,L2)), smaller(L1,L2,K1,K2).

%% Remove overlap
2{bad(peop(P1,P2)); bad(peop(Q1,Q2))}  :- atom(peop(P1,P2)), atom(peop(Q1,Q2)), P1 < Q1, P2 >= Q1.
:- ok(peop(P1,P2)), ok(peop(Q1,Q2)), P1 < Q1, P2 >= Q1.

2{bad(loc(P1,P2)); bad(loc(Q1,Q2))}  :- atom(loc(P1,P2)), atom(loc(Q1,Q2)), P1 < Q1, P2 >= Q1.
:- ok(loc(P1,P2)), ok(loc(Q1,Q2)), P1 < Q1, P2 >= Q1.

2{bad(org(P1,P2)); bad(org(Q1,Q2))}  :- atom(org(P1,P2)), atom(org(Q1,Q2)), P1 < Q1, P2 >= Q1.
:- ok(org(P1,P2)), ok(org(Q1,Q2)), P1 < Q1, P2 >= Q1.

2 {bad(other(P1,P2)); bad(other(Q1,Q2))}  :- atom(other(P1,P2)), atom(other(Q1,Q2)), P1 < Q1, P2 >= Q1.
:- ok(other(P1,P2)), ok(other(Q1,Q2)), P1 < Q1, P2 >= Q1.

%% Potentially incorrect

bad(liveIn(P1,P2,Q1,Q2)) :- atom(liveIn(P1,P2,Q1,Q2)), bad(peop(P1,P2)).
bad(liveIn(P1,P2,Q1,Q2)) :- atom(liveIn(P1,P2,Q1,Q2)), bad(loc(Q1,Q2)).

bad(locatedIn(P1,P2,Q1,Q2)) :- atom(locatedIn(P1,P2,Q1,Q2)), bad(loc(P1,P2)).
bad(locatedIn(P1,P2,Q1,Q2)) :- atom(locatedIn(P1,P2,Q1,Q2)), bad(loc(Q1,Q2)).

bad(orgbasedIn(P1,P2,Q1,Q2)) :- atom(orgbasedIn(P1,P2,Q1,Q2)), bad(org(P1,P2)).
bad(orgbasedIn(P1,P2,Q1,Q2)) :- atom(orgbasedIn(P1,P2,Q1,Q2)), bad(loc(Q1,Q2)).

bad(workFor(P1,P2,Q1,Q2)) :- atom(workFor(P1,P2,Q1,Q2)), bad(peop(P1,P2)).
bad(workFor(P1,P2,Q1,Q2)) :- atom(workFor(P1,P2,Q1,Q2)), bad(org(Q1,Q2)).

bad(kill(P1,P2,Q1,Q2)) :- atom(kill(P1,P2,Q1,Q2)), bad(peop(P1,P2)).
bad(kill(P1,P2,Q1,Q2)) :- atom(kill(P1,P2,Q1,Q2)), bad(peop(Q1,Q2)).

% Relation_exist constrain
relation_exist :- ok(liveIn(P1,P2,Q1,Q2)).
relation_exist :- ok(locatedIn(P1,P2,Q1,Q2)).
relation_exist :- ok(orgbasedIn(P1,P2,Q1,Q2)).
relation_exist :- ok(workFor(P1,P2,Q1,Q2)).
relation_exist :- ok(kill(P1,P2,Q1,Q2)).
:- not relation_exist.

