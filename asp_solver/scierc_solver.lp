% projection

atom(A) :- atom(A,P).
prob(A,P) :- atom(A,P).

prod(P, A) :- atom(A, P), ok(A).
invprod(P, A) :- atom(A, P), nok(A).

%% Notes:
%% - Entities: Task, Method, Material, OtherScientificTerm, Metric, Generic
%% - Relations: Used-for, Feature-of, Hyponym-of, Evaluate-for, Part-of, Compare(symmetric), Conjunction(symmetric)

%% A sub-word must be completely inside a word or completely outside of it
2{bad(entity(NE, P1, P2)); bad(entity(NE', Q1, Q2))} :- atom(entity(NE, P1, P2)),
									                    atom(entity(NE', Q1, Q2)),
									                    P1 < Q1, P2 > Q1, P2 < Q2.
:- ok(entity(NE,P1,P2)), ok(entity(NE',Q1,Q2)), P1 < Q1, P2 > Q1, P2 < Q2.

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


%% A sub-word pair (parent,child) should have identical types or one of the type is generic/otherScientificTerm
%% or parent's type is task.
%% (CH,C1,C2) is a sub-word of (PA,P1,P2)
2{bad(entity(CH, C1, C2)); bad(entity(PA, P1, P2))}:- atom(entity(CH, C1, C2)),
                                                        atom(entity(PA, P1, P2)),
                                                        C1 >= P1, C2 <= P2,
                                                        1 {C1!=P1; C2!=P2},
                                                        PA != CH,
                                                        PA != task,
                                                        PA != otherScientificTerm,
                                                        PA != generic,
                                                        CH != otherScientificTerm,
                                                        CH != generic.

:- ok(entity(CH, C1, C2)); ok(entity(PA, P1, P2)), atom(entity(CH, C1, C2)),
                                                atom(entity(PA, P1, P2)),
                                                C1 >= P1, C2 <= P2,
                                                1 {C1!=P1; C2!=P2},
                                                PA != CH,
                                                PA != task,
                                                PA != otherScientificTerm,
                                                PA != generic,
                                                CH != otherScientificTerm,
                                                CH != generic.


%% Eliminate relation with missing head or tail
:- ok(relation(NR,P,Q,P',Q')) , not ok(entity(_,P,Q)).
:- ok(relation(NR,P,Q,P',Q')) , not ok(entity(_,P',Q')).


%%%%%%%%%%% inference rules %%%%%%%%%%%%%%%%%%%%
%% - Relations: Used-for, Feature-of, Hyponym-of, Evaluate-for, Part-of, Compare(symmetric), Conjunction(symmetric)

rule(relation(useFor,P1,P2,Q1,Q2), relation(compare,O1,O2,P1,P2), relation(useFor,O1,O2,Q1,Q2)):-
	atom(relation(useFor,P1,P2,Q1,Q2)), atom(relation(compare,O1,O2,P1,P2)), P1 != P2, P1 != Q1.
rule(relation(useFor,Q1,Q2,P1,P2), relation(compare,O1,O2,P1,P2), relation(useFor,Q1,Q2,O1,O2)):-
	atom(relation(useFor,Q1,Q2,P1,P2)), atom(relation(compare,O1,O2,P1,P2)), P1 != P2, P1 != Q1.

rule(relation(featureOf,P1,P2,Q1,Q2), relation(compare,O1,O2,P1,P2), relation(featureOf,O1,O2,Q1,Q2)):-
	atom(relation(featureOf,P1,P2,Q1,Q2)), atom(relation(compare,O1,O2,P1,P2)), P1 != P2, P1 != Q1.
rule(relation(featureOf,Q1,Q2,P1,P2), relation(compare,O1,O2,P1,P2), relation(featureOf,Q1,Q2,O1,O2)):-
	atom(relation(featureOf,Q1,Q2,P1,P2)), atom(relation(compare,O1,O2,P1,P2)), P1 != P2, P1 != Q1.

rule(relation(evaluateFor,P1,P2,Q1,Q2), relation(compare,O1,O2,P1,P2), relation(evaluateFor,O1,O2,Q1,Q2)):-
	atom(relation(evaluateFor,P1,P2,Q1,Q2)), atom(relation(compare,O1,O2,P1,P2)), P1 != P2, P1 != Q1.
rule(relation(evaluateFor,Q1,Q2,P1,P2), relation(compare,O1,O2,P1,P2), relation(evaluateFor,Q1,Q2,O1,O2)):-
	atom(relation(evaluateFor,Q1,Q2,P1,P2)), atom(relation(compare,O1,O2,P1,P2)), P1 != P2, P1 != Q1.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rule(relation(useFor,P1,P2,Q1,Q2), relation(conjunction,O1,O2,P1,P2), relation(useFor,O1,O2,Q1,Q2)):-
	atom(relation(useFor,P1,P2,Q1,Q2)), atom(relation(conjunction,O1,O2,P1,P2)), P1 != P2, P1 != Q1.
rule(relation(useFor,Q1,Q2,P1,P2), relation(conjunction,O1,O2,P1,P2), relation(useFor,Q1,Q2,O1,O2)):-
	atom(relation(useFor,Q1,Q2,P1,P2)), atom(relation(conjunction,O1,O2,P1,P2)), P1 != P2, P1 != Q1.

rule(relation(featureOf,P1,P2,Q1,Q2), relation(conjunction,O1,O2,P1,P2), relation(featureOf,O1,O2,Q1,Q2)):-
	atom(relation(featureOf,P1,P2,Q1,Q2)), atom(relation(conjunction,O1,O2,P1,P2)), P1 != P2, P1 != Q1.
rule(relation(featureOf,Q1,Q2,P1,P2), relation(conjunction,O1,O2,P1,P2), relation(featureOf,Q1,Q2,O1,O2)):-
	atom(relation(featureOf,Q1,Q2,P1,P2)), atom(relation(conjunction,O1,O2,P1,P2)), P1 != P2, P1 != Q1.

rule(relation(evaluateFor,P1,P2,Q1,Q2), relation(conjunction,O1,O2,P1,P2), relation(evaluateFor,O1,O2,Q1,Q2)):-
	atom(relation(evaluateFor,P1,P2,Q1,Q2)), atom(relation(conjunction,O1,O2,P1,P2)), P1 != P2, P1 != Q1.
rule(relation(evaluateFor,Q1,Q2,P1,P2), relation(conjunction,O1,O2,P1,P2), relation(evaluateFor,Q1,Q2,O1,O2)):-
	atom(relation(evaluateFor,Q1,Q2,P1,P2)), atom(relation(conjunction,O1,O2,P1,P2)), P1 != P2, P1 != Q1.

rule(relation(hyponymOf,P1,P2,Q1,Q2), relation(conjunction,O1,O2,P1,P2), relation(hyponymOf,O1,O2,Q1,Q2)):-
	atom(relation(hyponymOf,P1,P2,Q1,Q2)), atom(relation(conjunction,O1,O2,P1,P2)), P1 != P2, P1 != Q1.


%%%%% ASP for working with inference rules %%%%%%%%%%%%%%%%

6 {bad(X); bad(Y); bad(Z); inf_atom(Z);
    dependency(X, Z); dependency(Y, Z) } :-
       	rule(X, Y, Z), atom(X), atom(Y), not atom(Z).

:- ok(Y), inf_atom(Y), dependency(X,Y), not ok(X).
:- rule(X, Y, Z), ok(X), ok(Y), not ok(Z).


%For Compare and Conjunction: compare abbreviation with original concept, usually abbreviation is written first (cond1),
% sometimes written latter but in generic form (those, that,...)
% and the head (abbreviation) is shorter than the tail (original concept)

:- ok(relation(conjunction,P,Q,P',Q')), ok(entity(E,P,Q)), ok(entity(E',P',Q')),
    P > P', (Q-P) < (Q'-P'), E != generic, E != otherScientificTerm, E != E'.