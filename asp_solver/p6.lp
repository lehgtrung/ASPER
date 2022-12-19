% projection

atom(A) :- atom(A,P).
prob(A,P) :- atom(A,P).

prod(P) :- atom(A, P), ok(A).
invprod(P) :- atom(A, P), nok(A).

% Type requirements -- if the body is satisfied then one of the atom in the body is incorrect

2{bad(locatedIn(P,Q)); bad(org(P))}:- atom(locatedIn(P,Q)), atom(org(P)).
2{bad(locatedIn(P,Q)); bad(peop(P))} :- atom(locatedIn(P,Q)), atom(peop(P)).
2{bad(locatedIn(P,Q)); bad(other(P))} :- atom(locatedIn(P,Q)), atom(other(P)).
2{bad(locatedIn(P,Q)); bad(org(Q))}:- atom(locatedIn(P,Q)), atom(org(Q)).
2{bad(locatedIn(P,Q)); bad(peop(Q))} :- atom(locatedIn(P,Q)), atom(peop(Q)).
2{bad(locatedIn(P,Q)); bad(other(Q))}:- atom(locatedIn(P,Q)), atom(other(Q)).

2{bad(orgbasedIn(P,Q));bad(loc(P))}:- atom(orgbasedIn(P,Q)), atom(loc(P)).
2{bad(orgbasedIn(P,Q)); bad(peop(P))} :- atom(orgbasedIn(P,Q)), atom(peop(P)).
2{bad(orgbasedIn(P,Q)); bad(other(P))}:- atom(orgbasedIn(P,Q)), atom(other(P)).
2{bad(orgbasedIn(P,Q)); bad(org(Q))}:- atom(orgbasedIn(P,Q)), atom(org(Q)).
2{bad(orgbasedIn(P,Q)); bad(peop(Q))} :- atom(orgbasedIn(P,Q)), atom(peop(Q)).
2{bad(orgbasedIn(P,Q)); bad(other(Q))}:- atom(orgbasedIn(P,Q)), atom(other(Q)).

2{bad(liveIn(P,Q));bad(loc(P))}:- atom(liveIn(P,Q)), atom(loc(P)).
2{bad(liveIn(P,Q)); bad(org(P))}:- atom(liveIn(P,Q)), atom(org(P)).
2{bad(liveIn(P,Q)); bad(other(P))}:- atom(liveIn(P,Q)), atom(other(P)).
2{bad(liveIn(P,Q)); bad(peop(Q))} :- atom(liveIn(P,Q)), atom(peop(Q)).
2{bad(liveIn(P,Q)); bad(org(Q))}:- atom(liveIn(P,Q)), atom(org(Q)).
2{bad(liveIn(P,Q)); bad(other(Q))}:- atom(liveIn(P,Q)), atom(other(Q)).

2{bad(workFor(P,Q));bad(loc(P))}:- atom(workFor(P,Q)), atom(loc(P)).
2{bad(workFor(P,Q)); bad(org(P))}:- atom(workFor(P,Q)), atom(org(P)).
2{bad(workFor(P,Q)); bad(other(P))}:- atom(workFor(P,Q)), atom(other(P)).
2{bad(workFor(P,Q)); bad(peop(Q))} :- atom(workFor(P,Q)), atom(peop(Q)).
2{bad(workFor(P,Q));bad(loc(Q))}:- atom(workFor(P,Q)), atom(loc(Q)).
2{bad(workFor(P,Q)); bad(other(Q))}:- atom(workFor(P,Q)), atom(other(Q)).

2{bad(kill(P,Q));bad(loc(P))}:- atom(kill(P,Q)), atom(loc(P)).
2{bad(kill(P,Q)); bad(org(P))}:- atom(kill(P,Q)), atom(org(P)).
2{bad(kill(P,Q)); bad(other(P))}:- atom(kill(P,Q)), atom(other(P)).
2{bad(kill(P,Q));bad(loc(Q))}:- atom(kill(P,Q)), atom(loc(Q)).
2{bad(kill(P,Q));bad(org(Q))}:- atom(kill(P,Q)), atom(org(Q)).
2{bad(kill(P,Q)); bad(other(Q))}:- atom(kill(P,Q)), atom(other(Q)).


% Other requirements
bad(liveIn(X,X)):- atom(liveIn(X,X)).
bad(locatedIn(X,X)):- atom(locatedIn(X,X)).
bad(orgbasedIn(X,X)):- atom(orgbasedIn(X,X)).
bad(workFor(X,X)):- atom(workFor(X,X)).
bad(kill(X,X)):- atom(kill(X,X)).

2{bad(peop(X)); bad(loc(X))} :- atom(peop(X)), atom(loc(X)).
2{bad(peop(X)); bad(org(X))} :- atom(peop(X)), atom(org(X)).
2{bad(peop(X)); bad(other(X))}:- atom(peop(X)), atom(other(X)).
2{bad(loc(X)); bad(org(X))}:- atom(loc(X)), atom(org(X)).
2{bad(loc(X)); bad(other(X))}:- atom(loc(X)), atom(other(X)).
2{bad(org(X)); bad(other(X))}:- atom(org(X)), atom(other(X)).

% Type requirements --
% the body of a constraint should not be satisfied

:- ok(locatedIn(P,Q)), ok(org(P)).
:- ok(locatedIn(P,Q)), ok(peop(P)).
:- ok(locatedIn(P,Q)), ok(other(P)).
:- ok(locatedIn(P,Q)), ok(org(Q)).
:- ok(locatedIn(P,Q)), ok(peop(Q)).
:- ok(locatedIn(P,Q)), ok(other(Q)).

:- ok(orgbasedIn(P,Q)), ok(loc(P)).
:- ok(orgbasedIn(P,Q)), ok(peop(P)).
:- ok(orgbasedIn(P,Q)), ok(other(P)).
:- ok(orgbasedIn(P,Q)), ok(org(Q)).
:- ok(orgbasedIn(P,Q)), ok(peop(Q)).
:- ok(orgbasedIn(P,Q)), ok(other(Q)).

:- ok(liveIn(P,Q)), ok(loc(P)).
:- ok(liveIn(P,Q)), ok(org(P)).
:- ok(liveIn(P,Q)), ok(other(P)).
:- ok(liveIn(P,Q)), ok(peop(Q)).
:- ok(liveIn(P,Q)), ok(org(Q)).
:- ok(liveIn(P,Q)), ok(other(Q)).

:- ok(workFor(P,Q)), ok(loc(P)).
:- ok(workFor(P,Q)), ok(org(P)).
:- ok(workFor(P,Q)), ok(other(P)).
:- ok(workFor(P,Q)), ok(peop(Q)).
:- ok(workFor(P,Q)), ok(loc(Q)).
:- ok(workFor(P,Q)), ok(other(Q)).

:- ok(kill(P,Q)), ok(loc(P)).
:- ok(kill(P,Q)), ok(org(P)).
:- ok(kill(P,Q)), ok(other(P)).
:- ok(kill(P,Q)), ok(loc(Q)).
:- ok(kill(P,Q)), ok(org(Q)).
:- ok(kill(P,Q)), ok(other(Q)).

%% Other requirements

:- ok(liveIn(X,X)).
:- ok(locatedIn(X,X)).
:- ok(orgbasedIn(X,X)).
:- ok(workFor(X,X)).
:- ok(kill(X,X)).

:- ok(peop(X)), ok(loc(X)).
:- ok(peop(X)), ok(org(X)).
:- ok(peop(X)), ok(other(X)).
:- ok(loc(X)), ok(org(X)).
:- ok(loc(X)), ok(other(X)).
:- ok(org(X)), ok(other(X)).





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

rule(locatedIn(P1,P2), orgbasedIn(O,P1), orgbasedIn(O, P2)):-
	atom(locatedIn(P1,P2)), atom(orgbasedIn(O,P1)), P1 != P2.

rule(locatedIn(P1,P2), locatedIn(P2,P3), locatedIn(P1, P3)):-
	atom(locatedIn(P1,P2)), atom(locatedIn(P2,P3)), P1 != P2, P2 != P3.

rule(liveIn(X,P1), locatedIn(P1,P2), liveIn(X,P2)):-
	atom(liveIn(X,P1)), atom(locatedIn(P1,P2)), P1!=P2.

rule(workFor(X,O), orgbasedIn(O,P),liveIn(X,P)):-
	atom(workFor(X,O)), atom(orgbasedIn(O,P)).

rule(liveIn(X,P), workFor(X,O), orgbasedIn(O,P)):-
	atom(liveIn(X,P)), atom(workFor(X,O)).



%%%%% ASP for working with inference rules %%%%%%%%%%%%%%%%

6 {bad(X); bad(Y); bad(Z); inf_atom(Z);
    dependency(X, Z); dependency(Y, Z) } :-
       	rule(X, Y, Z), atom(X), atom(Y), not atom(Z).

:- ok(Y), inf_atom(Y), dependency(X,Y), not ok(X).
:- rule(X, Y, Z), ok(X), ok(Y), not ok(Z).



%% Trung: infer types for relations after they are selected
%% locatedIn/orgBasedIn/liveIn/workFor/kill
ok(loc(P1)) :- ok(locatedIn(P1,P2)).
ok(loc(P2)) :- ok(locatedIn(P1,P2)).
ok(org(O)) :- ok(orgBasedIn(O,P)).
ok(loc(P)) :- ok(orgBasedIn(O,P)).
ok(peop(X)) :- ok(liveIn(X,P)).
ok(loc(P)) :- ok(liveIn(X,P)).
ok(peop(X)) :- ok(workFor(X,O)).
ok(org(O)) :- ok(workFor(X,O)).
ok(peop(X)) :- ok(kill(X,Y)).
ok(peop(Y)) :- ok(kill(X,Y)).


