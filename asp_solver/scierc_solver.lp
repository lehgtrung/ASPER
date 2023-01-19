% projection

atom(A) :- atom(A,P).
prob(A,P) :- atom(A,P).

prod(P, A) :- atom(A, P), ok(A).
invprod(P, A) :- atom(A, P), nok(A).

%% Notes:
%% - Entities: Task, Method, Material, OtherScientificTerm, Metric, Generic
%% - Relations: Used-for, Feature-of, Hyponym-of, Evaluate-for, Part-of, Compare(symmetric), Conjunction(symmetric)

%% overlapping
2{bad(entity(NE, P1, P2)); bad(entity(NE', Q1, Q2))} :- atom(entity(NE, P1, P2)),
									  atom(entity(NE', Q1, Q2)),
									  P1 < Q1, P2 > Q1.
:- ok(entity(NE,P1,P2)), ok(entity(NE',Q1,Q2)), P1 < Q1, P2 > Q1.

2{bad(entity(NE, P, Q)); bad(entity(NE', P, R))} :- atom(entity(NE, P, Q)),
									  atom(entity(NE', P, R)), R != Q.
:- ok(entity(NE, P, Q)), ok(entity(NE', P, R)), R != Q.

%% Eliminate relation with missing head or tail
%:- ok(relation(NR,P,Q,P',Q')) , not ok(entity(_,P,Q)).
%:- ok(relation(NR,P,Q,P',Q')) , not ok(entity(_,P',Q')).


%% Eliminate reciprocal relations
2{bad(relation(NR,P,Q,P',Q')); bad(relation(NT,P',Q',P,Q))} :- atom(relation(NR,P,Q,P',Q')), atom(relation(NT,P',Q',P,Q)).
:- ok(relation(NR,P,Q,P',Q')), ok(relation(NT,P',Q',P,Q)).


%%% Relation exists
relation_flag.
relation_exist :- ok(relation(NR, P, Q, P', Q')).
% :- not relation_exist.


%%%%%%%%%%% inference rules %%%%%%%%%%%%%%%%%%%%
%% - Relations: Used-for, Feature-of, Hyponym-of, Evaluate-for, Part-of, Compare(symmetric), Conjunction(symmetric)

rule(relation(usedFor,P1,P2,Q1,Q2), relation(compare,O1,O2,P1,P2), relation(usedFor,O1,O2,Q1,Q2)):-
	atom(relation(usedFor,P1,P2,Q1,Q2)), atom(relation(compare,O1,O2,P1,P2)), not bad(entity(_,O1,O2)).
rule(relation(usedFor,Q1,Q2,P1,P2), relation(compare,O1,O2,P1,P2), relation(usedFor,Q1,Q2,O1,O2)):-
	atom(relation(usedFor,Q1,Q2,P1,P2)), atom(relation(compare,O1,O2,P1,P2)), not bad(entity(_,O1,O2)).
rule(relation(usedFor,P1,P2,Q1,Q2), relation(compare,O1,O2,P1,P2), relation(usedFor,O1,O2,Q1,Q2)):-
	atom(relation(usedFor,P1,P2,Q1,Q2)), atom(relation(compare,O1,O2,P1,P2)), not bad(entity(_,P1,P2)).
rule(relation(usedFor,Q1,Q2,P1,P2), relation(compare,O1,O2,P1,P2), relation(usedFor,Q1,Q2,O1,O2)):-
	atom(relation(usedFor,Q1,Q2,P1,P2)), atom(relation(compare,O1,O2,P1,P2)), not bad(entity(_,P1,P2)).


rule(relation(featureOf,P1,P2,Q1,Q2), relation(compare,O1,O2,P1,P2), relation(featureOf,O1,O2,Q1,Q2)):-
	atom(relation(featureOf,P1,P2,Q1,Q2)), atom(relation(compare,O1,O2,P1,P2)), not bad(entity(_,O1,O2)).
rule(relation(featureOf,Q1,Q2,P1,P2), relation(compare,O1,O2,P1,P2), relation(featureOf,Q1,Q2,O1,O2)):-
	atom(relation(featureOf,Q1,Q2,P1,P2)), atom(relation(compare,O1,O2,P1,P2)), not bad(entity(_,O1,O2)).
rule(relation(featureOf,P1,P2,Q1,Q2), relation(compare,O1,O2,P1,P2), relation(featureOf,O1,O2,Q1,Q2)):-
	atom(relation(featureOf,P1,P2,Q1,Q2)), atom(relation(compare,O1,O2,P1,P2)), not bad(entity(_,P1,P2)).
rule(relation(featureOf,Q1,Q2,P1,P2), relation(compare,O1,O2,P1,P2), relation(featureOf,Q1,Q2,O1,O2)):-
	atom(relation(featureOf,Q1,Q2,P1,P2)), atom(relation(compare,O1,O2,P1,P2)), not bad(entity(_,P1,P2)).

rule(relation(evaluateFor,P1,P2,Q1,Q2), relation(compare,O1,O2,P1,P2), relation(evaluateFor,O1,O2,Q1,Q2)):-
	atom(relation(evaluateFor,P1,P2,Q1,Q2)), atom(relation(compare,O1,O2,P1,P2)), not bad(entity(_,O1,O2)).
rule(relation(evaluateFor,Q1,Q2,P1,P2), relation(compare,O1,O2,P1,P2), relation(evaluateFor,Q1,Q2,O1,O2)):-
	atom(relation(evaluateFor,Q1,Q2,P1,P2)), atom(relation(compare,O1,O2,P1,P2)), not bad(entity(_,O1,O2)).
rule(relation(evaluateFor,P1,P2,Q1,Q2), relation(compare,O1,O2,P1,P2), relation(evaluateFor,O1,O2,Q1,Q2)):-
	atom(relation(evaluateFor,P1,P2,Q1,Q2)), atom(relation(compare,O1,O2,P1,P2)), not bad(entity(_,P1,P2)).
rule(relation(evaluateFor,Q1,Q2,P1,P2), relation(compare,O1,O2,P1,P2), relation(evaluateFor,Q1,Q2,O1,O2)):-
	atom(relation(evaluateFor,Q1,Q2,P1,P2)), atom(relation(compare,O1,O2,P1,P2)), not bad(entity(_,P1,P2)).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rule(relation(usedFor,P1,P2,Q1,Q2), relation(conjunction,O1,O2,P1,P2), relation(usedFor,O1,O2,Q1,Q2)):-
	atom(relation(usedFor,P1,P2,Q1,Q2)), atom(relation(conjunction,O1,O2,P1,P2)), not bad(entity(_,O1,O2)).
rule(relation(usedFor,Q1,Q2,P1,P2), relation(conjunction,O1,O2,P1,P2), relation(usedFor,Q1,Q2,O1,O2)):-
	atom(relation(usedFor,Q1,Q2,P1,P2)), atom(relation(conjunction,O1,O2,P1,P2)), not bad(entity(_,O1,O2)).
rule(relation(usedFor,P1,P2,Q1,Q2), relation(conjunction,O1,O2,P1,P2), relation(usedFor,O1,O2,Q1,Q2)):-
	atom(relation(usedFor,P1,P2,Q1,Q2)), atom(relation(conjunction,O1,O2,P1,P2)), not bad(entity(_,P1,P2)).
rule(relation(usedFor,Q1,Q2,P1,P2), relation(conjunction,O1,O2,P1,P2), relation(usedFor,Q1,Q2,O1,O2)):-
	atom(relation(usedFor,Q1,Q2,P1,P2)), atom(relation(conjunction,O1,O2,P1,P2)), not bad(entity(_,P1,P2)).

rule(relation(featureOf,P1,P2,Q1,Q2), relation(conjunction,O1,O2,P1,P2), relation(featureOf,O1,O2,Q1,Q2)):-
	atom(relation(featureOf,P1,P2,Q1,Q2)), atom(relation(conjunction,O1,O2,P1,P2)), not bad(entity(_,O1,O2)).
rule(relation(featureOf,Q1,Q2,P1,P2), relation(conjunction,O1,O2,P1,P2), relation(featureOf,Q1,Q2,O1,O2)):-
	atom(relation(featureOf,Q1,Q2,P1,P2)), atom(relation(conjunction,O1,O2,P1,P2)), not bad(entity(_,O1,O2)).
rule(relation(featureOf,P1,P2,Q1,Q2), relation(conjunction,O1,O2,P1,P2), relation(featureOf,O1,O2,Q1,Q2)):-
	atom(relation(featureOf,P1,P2,Q1,Q2)), atom(relation(conjunction,O1,O2,P1,P2)), not bad(entity(_,P1,P2)).
rule(relation(featureOf,Q1,Q2,P1,P2), relation(conjunction,O1,O2,P1,P2), relation(featureOf,Q1,Q2,O1,O2)):-
	atom(relation(featureOf,Q1,Q2,P1,P2)), atom(relation(conjunction,O1,O2,P1,P2)), not bad(entity(_,P1,P2)).

rule(relation(evaluateFor,P1,P2,Q1,Q2), relation(conjunction,O1,O2,P1,P2), relation(evaluateFor,O1,O2,Q1,Q2)):-
	atom(relation(evaluateFor,P1,P2,Q1,Q2)), atom(relation(conjunction,O1,O2,P1,P2)), not bad(entity(_,O1,O2)).
rule(relation(evaluateFor,Q1,Q2,P1,P2), relation(conjunction,O1,O2,P1,P2), relation(evaluateFor,Q1,Q2,O1,O2)):-
	atom(relation(evaluateFor,Q1,Q2,P1,P2)), atom(relation(conjunction,O1,O2,P1,P2)), not bad(entity(_,O1,O2)).
rule(relation(evaluateFor,P1,P2,Q1,Q2), relation(conjunction,O1,O2,P1,P2), relation(evaluateFor,O1,O2,Q1,Q2)):-
	atom(relation(evaluateFor,P1,P2,Q1,Q2)), atom(relation(conjunction,O1,O2,P1,P2)), not bad(entity(_,P1,P2)).
rule(relation(evaluateFor,Q1,Q2,P1,P2), relation(conjunction,O1,O2,P1,P2), relation(evaluateFor,Q1,Q2,O1,O2)):-
	atom(relation(evaluateFor,Q1,Q2,P1,P2)), atom(relation(conjunction,O1,O2,P1,P2)), not bad(entity(_,P1,P2)).

rule(relation(hyponymOf,P1,P2,Q1,Q2), relation(conjunction,O1,O2,P1,P2), relation(hyponymOf,O1,O2,Q1,Q2)):-
	atom(relation(hyponymOf,P1,P2,Q1,Q2)), atom(relation(conjunction,O1,O2,P1,P2)), not bad(entity(_,O1,O2)).
rule(relation(hyponymOf,P1,P2,Q1,Q2), relation(conjunction,O1,O2,P1,P2), relation(hyponymOf,O1,O2,Q1,Q2)):-
	atom(relation(hyponymOf,P1,P2,Q1,Q2)), atom(relation(conjunction,O1,O2,P1,P2)), not bad(entity(_,P1,P2)).


%%%%% ASP for working with inference rules %%%%%%%%%%%%%%%%

6 {bad(X); bad(Y); bad(Z); inf_atom(Z);
    dependency(X, Z); dependency(Y, Z) } :-
       	rule(X, Y, Z), atom(X), atom(Y), not atom(Z).

:- ok(Y), inf_atom(Y), dependency(X,Y), not ok(X).
:- rule(X, Y, Z), ok(X), ok(Y), not ok(Z).


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

#show ok/1.

