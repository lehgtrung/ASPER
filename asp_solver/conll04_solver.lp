% projection

atom(A) :- atom(A,P).
prob(A,P) :- atom(A,P).

prod(P, A) :- atom(A, P), ok(A).
invprod(P, A) :- atom(A, P), nok(A).

%%% representation
%     entity(name, P, Q) 			e.g.  entity(peop, 1, 3) instead of peop(1,3)
%     relation(name, P, Q, P', Q') 		e.g. 	relation(kill, 1, 3, 5, 7)
%     type_def(relation, type1, type2)	e.g.	type_def(kill, peop, peop)
%%%
%% %
%%%
%     Reasoning about type:

%% Type def for CoNLL04
type_def(liveIn, peop, loc).
type_def(locatedIn, loc, loc).
type_def(orgbasedIn, org, loc).
type_def(workFor, peop, org).
type_def(kill, peop, peop).


2{bad(relation(NR, P, Q, P', Q')); bad(entity(NE, P, Q))} :- 	atom(relation(NR, P, Q, P', Q')),
											atom(entity(NE, P, Q)),
											type_def(NR, E1, E2),
											E1 != NE.

2{bad(relation(NR, P, Q, P', Q')); bad(entity(NE, P', Q'))} :- atom(relation(NR, P, Q, P', Q')),
											atom(entity(NE, P', Q')),
											type_def(NR, E1, E2),
											E2 != NE.

%% overlapping

2{bad(entity(NE, P1, P2)); bad(entity(NE', Q1, Q2))} :- atom(entity(NE, P1, P2)),
									  atom(entity(NE', Q1, Q2)),
									  P1 < Q1, P2 > Q1.
:- ok(entity(NE,P1,P2)), ok(entity(NE',Q1,Q2)), P1 < Q1, P2 > Q1.

%% potentially incorrect

bad(relation(NR, P, Q, P', Q')) :- atom(relation(NR, P, Q, P', Q')),
					 	 type_def(NR, E, E'),
						 bad(entity(E, P, Q)).

bad(relation(NR, P, Q, P', Q')) :- atom(relation(NR, P, Q, P', Q')),
					 	 type_def(NR, E, E'),
						 bad(entity(E', P', Q')).

:- ok(relation(NR, P, Q, P', Q')), ok(entity(NE, P, Q)), type_def(NR, NE', _), NE != NE'.
:- ok(relation(NR, P, Q, P', Q')), ok(entity(NE, P', Q')), type_def(NR, _, NE'), NE != NE'.

:- ok(relation(_, P, Q, P, Q)).

ok(entity(E, P, Q)) :- ok(relation(NR, P, Q, P', Q')), type_def(NR, E, E').
ok(entity(E', P', Q')) :- ok(relation(NR, P, Q, P', Q')), type_def(NR, E, E').


%%% Relation exists
relation_exist :- ok(relation(NR, P, Q, P', Q')).
% :- not relation_exist.

%%%%%%%%%%% inference rules %%%%%%%%%%%%%%%%%%%%

rule(relation(locatedIn,P1,P2,Q1,Q2), relation(orgbasedIn,O1,O2,P1,P2), relation(orgbasedIn,O1,O2,Q1,Q2)):-
	atom(relation(locatedIn,P1,P2,Q1,Q2)), atom(relation(orgbasedIn,O1,O2,P1,P2)), P1 != P2, P1 != Q1.

rule(relation(locatedIn,P1,P2,Q1,Q2), relation(locatedIn,Q1,Q2,R1,R2), relation(locatedIn,P1,P2,R1,R2)):-
	atom(relation(locatedIn,P1,P2,Q1,Q2)), atom(relation(locatedIn,Q1,Q2,R1,R2)), P1 != P2, P1 != Q1, Q1 != Q2, Q1 != R1.

rule(relation(liveIn,X1,X2,P1,P2), relation(locatedIn,P1,P2,Q1,Q2), relation(liveIn,X1,X2,Q1,Q2)):-
	atom(relation(liveIn,X1,X2,P1,P2)), atom(relation(locatedIn,P1,P2,Q1,Q2)), P1 != P2, P1 != Q1.


%%%%% ASP for working with inference rules %%%%%%%%%%%%%%%%

6 {bad(X); bad(Y); bad(Z); inf_atom(Z);
    dependency(X, Z); dependency(Y, Z) } :-
       	rule(X, Y, Z), atom(X), atom(Y), not atom(Z).

:- ok(Y), inf_atom(Y), dependency(X,Y), not ok(X).
:- rule(X, Y, Z), ok(X), ok(Y), not ok(Z).

%% More rules %%
%% If someone is killed, he can't kill anyone else
dead(Q1,Q2) :- atom(relation(kill,P1,P2,Q1,Q2)).
{bad(relation(kill,Q1,Q2,R1,R2))} :- atom(relation(kill,Q1,Q2,R1,R2)), dead(Q1,Q2).
:- ok(relation(kill,Q1,Q2,R1,R2)), atom(relation(kill,Q1,Q2,R1,R2)), dead(Q1,Q2).

%% Reason about location's size
smaller(L1,L2,K1,K2) :- atom(relation(locatedIn,L1,L2,K1,K2)).
smaller(L1,L2,J1,J2) :- smaller(L1,L2,K1,K2), smaller(K1,K2,J1,J2).
{bad(relation(locatedIn,K1,K2,L1,L2))} :- smaller(L1,L2,K1,K2).
:- ok(relation(locatedIn,K1,K2,L1,L2)), smaller(L1,L2,K1,K2).


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

