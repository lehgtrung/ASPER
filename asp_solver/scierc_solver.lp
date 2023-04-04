% projection

atom(A) :- atom(A,P).
prob(A,P) :- atom(A,P).

prod(P, A) :- atom(A, P), ok(A).
invprod(P, A) :- atom(A, P), nok(A).

%% Notes:
%% - Entities: Task, Method, Material, OtherScientificTerm, Metric, Generic
%% - Relations: Used-for, Feature-of, Hyponym-of, Evaluate-for, Part-of, Compare(symmetric), Conjunction(symmetric)

%% overlapping
%2{bad(entity(NE, P1, P2)); bad(entity(NE', Q1, Q2))} :- atom(entity(NE, P1, P2)),
%									  atom(entity(NE', Q1, Q2)),
%									  P1 < Q1, P2 > Q1, P2 != Q2.
%:- ok(entity(NE,P1,P2)), ok(entity(NE',Q1,Q2)), P1 < Q1, P2 > Q1, P2 != Q2.

%2{bad(entity(NE, P, Q)); bad(entity(NE', P, R))} :- atom(entity(NE, P, Q)),
%									  atom(entity(NE', P, R)), R != Q.
%:- ok(entity(NE, P, Q)), ok(entity(NE', P, R)), R != Q.


%% Eliminate relation with missing head or tail
%% Solution: use model's prediction to infer head/tail type
exists(P,Q) :- ok(entity(_,P,Q)).
bad(relation(NR,P,Q,P',Q')) :- atom(relation(NR,P,Q,P',Q')) ,  not exists(P,Q).
bad(relation(NR,P,Q,P',Q')) :- atom(relation(NR,P,Q,P',Q')) ,  not exists(P',Q').
:- ok(relation(NR,P,Q,P',Q')) , {ok(entity(_,P,Q)); ok(entity(_,P',Q'))} 1.


%% Eliminate reciprocal relations
2{bad(relation(NR,P,Q,P',Q')); bad(relation(NT,P',Q',P,Q))} :- atom(relation(NR,P,Q,P',Q')),
                                                               atom(relation(NT,P',Q',P,Q)),
                                                               NT != NR,
                                                               P != P',
                                                               P != Q.
:- ok(relation(NR,P,Q,P',Q')), ok(relation(NT,P',Q',P,Q)), NT != NR,
                                                           P != P',
                                                           P != Q.

%%% Relation exists
relation_flag.
relation_exist :- ok(relation(NR, P, Q, P', Q')).
% :- not relation_exist.


%%%%%%%%%%% inference rules %%%%%%%%%%%%%%%%%%%%
% usedFor
%rule(relation(usedFor,P,P',R,R'), relation(conjunction,P,P',Q,Q'), relation(usedFor,Q,Q',R,R')):-
%	atom(relation(usedFor,P,P',R,R')), atom(relation(conjunction,P,P',Q,Q')), P != P', P != Q.
%rule(relation(usedFor,Q,Q',R,R'), relation(conjunction,P,P',Q,Q'), relation(usedFor,P,P',R,R')):-
%	atom(relation(usedFor,Q,Q',R,R')), atom(relation(conjunction,P,P',Q,Q')), P != P', P != Q.

%rule(relation(usedFor,R,R',P,P'), relation(conjunction,P,P',Q,Q'), relation(usedFor,R,R',Q,Q')):-
%	atom(relation(usedFor,R,R',P,P')), atom(relation(conjunction,P,P',Q,Q')), P != P', P != Q.
%rule(relation(usedFor,R,R',Q,Q'), relation(conjunction,P,P',Q,Q'), relation(usedFor,R,R',P,P')):-
%	atom(relation(usedFor,R,R',Q,Q')), atom(relation(conjunction,P,P',Q,Q')), P != P', P != Q.

% featureOf
%rule(relation(featureOf,P,P',R,R'), relation(conjunction,P,P',Q,Q'), relation(featureOf,Q,Q',R,R')):-
%	atom(relation(featureOf,P,P',R,R')), atom(relation(conjunction,P,P',Q,Q')), P != P', P != Q.
%rule(relation(featureOf,Q,Q',R,R'), relation(conjunction,P,P',Q,Q'), relation(featureOf,P,P',R,R')):-
%	atom(relation(featureOf,Q,Q',R,R')), atom(relation(conjunction,P,P',Q,Q')), P != P', P != Q.

%rule(relation(featureOf,R,R',P,P'), relation(conjunction,P,P',Q,Q'), relation(featureOf,R,R',Q,Q')):-
%	atom(relation(featureOf,R,R',P,P')), atom(relation(conjunction,P,P',Q,Q')), P != P', P != Q.
%rule(relation(featureOf,R,R',Q,Q'), relation(conjunction,P,P',Q,Q'), relation(featureOf,R,R',P,P')):-
%	atom(relation(featureOf,R,R',Q,Q')), atom(relation(conjunction,P,P',Q,Q')), P != P', P != Q.

% hyponymOf (good rule)
rule(relation(hyponymOf,P,P',R,R'), relation(conjunction,P,P',Q,Q'), relation(hyponymOf,Q,Q',R,R')):-
	atom(relation(hyponymOf,P,P',R,R')), atom(relation(conjunction,P,P',Q,Q')), P != P', P != Q.
rule(relation(hyponymOf,Q,Q',R,R'), relation(conjunction,P,P',Q,Q'), relation(hyponymOf,P,P',R,R')):-
	atom(relation(hyponymOf,Q,Q',R,R')), atom(relation(conjunction,P,P',Q,Q')), P != P', P != Q.

rule(relation(hyponymOf,R,R',P,P'), relation(conjunction,P,P',Q,Q'), relation(hyponymOf,R,R',Q,Q')):-
	atom(relation(hyponymOf,R,R',P,P')), atom(relation(conjunction,P,P',Q,Q')), P != P', P != Q.
rule(relation(hyponymOf,R,R',Q,Q'), relation(conjunction,P,P',Q,Q'), relation(hyponymOf,R,R',P,P')):-
	atom(relation(hyponymOf,R,R',Q,Q')), atom(relation(conjunction,P,P',Q,Q')), P != P', P != Q.

% evaluateFor (good rule)
rule(relation(evaluateFor,P,P',R,R'), relation(conjunction,P,P',Q,Q'), relation(evaluateFor,Q,Q',R,R')):-
	atom(relation(evaluateFor,P,P',R,R')), atom(relation(conjunction,P,P',Q,Q')), P != P', P != Q.
rule(relation(evaluateFor,Q,Q',R,R'), relation(conjunction,P,P',Q,Q'), relation(evaluateFor,P,P',R,R')):-
	atom(relation(evaluateFor,Q,Q',R,R')), atom(relation(conjunction,P,P',Q,Q')), P != P', P != Q.

rule(relation(evaluateFor,R,R',P,P'), relation(conjunction,P,P',Q,Q'), relation(evaluateFor,R,R',Q,Q')):-
	atom(relation(evaluateFor,R,R',P,P')), atom(relation(conjunction,P,P',Q,Q')), P != P', P != Q.
rule(relation(evaluateFor,R,R',Q,Q'), relation(conjunction,P,P',Q,Q'), relation(evaluateFor,R,R',P,P')):-
	atom(relation(evaluateFor,R,R',Q,Q')), atom(relation(conjunction,P,P',Q,Q')), P != P', P != Q.

% partOf
%rule(relation(partOf,P,P',R,R'), relation(conjunction,P,P',Q,Q'), relation(partOf,Q,Q',R,R')):-
%	atom(relation(partOf,P,P',R,R')), atom(relation(conjunction,P,P',Q,Q')), P != P', P != Q.
%rule(relation(partOf,Q,Q',R,R'), relation(conjunction,P,P',Q,Q'), relation(partOf,P,P',R,R')):-
%	atom(relation(partOf,Q,Q',R,R')), atom(relation(conjunction,P,P',Q,Q')), P != P', P != Q.

%rule(relation(partOf,R,R',P,P'), relation(conjunction,P,P',Q,Q'), relation(partOf,R,R',Q,Q')):-
%	atom(relation(partOf,R,R',P,P')), atom(relation(conjunction,P,P',Q,Q')), P != P', P != Q.
%rule(relation(partOf,R,R',Q,Q'), relation(conjunction,P,P',Q,Q'), relation(partOf,R,R',P,P')):-
%	atom(relation(partOf,R,R',Q,Q')), atom(relation(conjunction,P,P',Q,Q')), P != P', P != Q.

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

