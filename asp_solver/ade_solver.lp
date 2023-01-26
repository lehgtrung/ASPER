% projection

atom(A) :- atom(A,P).
prob(A,P) :- atom(A,P).

prod(P, A) :- atom(A, P), ok(A).
invprod(P, A) :- atom(A, P), nok(A).

%% Notes:
%% - Entities: Adverse-Effect, Drug
%% - Relations: Adverse-Effect (Adverse-Effect to a Drug)

%% Type def for CoNLL04
type_def(adverseEffect, adverseEffect, drug).

%overlap_typedef(drug, adverseEffect).

%% overlapping
2{bad(entity(NE, P1, P2)); bad(entity(NE', Q1, Q2))} :- atom(entity(NE, P1, P2)),
									  atom(entity(NE', Q1, Q2)),
									  P1 < Q1, P2 > Q1.
:- ok(entity(NE,P1,P2)), ok(entity(NE',Q1,Q2)), P1 < Q1, P2 > Q1.

2{bad(entity(NE, P, Q)); bad(entity(NE', P, R))} :- atom(entity(NE, P, Q)),
									  atom(entity(NE', P, R)), R != Q.
:- ok(entity(NE, P, Q)), ok(entity(NE', P, R)), R != Q.


%2{bad(entity(CH, C1, C2)); bad(entity(PA, P1, P2))} :- atom(entity(CH, C1, C2)),
%                                                       atom(entity(PA, P1, P2)),
%                                                       C1 >= P1, C2 <= P2,
%                                                       1 {C1!=P1; C2!=P2},
%                                                       not overlap_typedef(CH,PA).

%:- ok(entity(CH, C1, C2)), ok(entity(PA, P1, P2)), C1 >= P1, C2 <= P2,
%                                                   1 {C1!=P1; C2!=P2},
%                                                   not overlap_typedef(CH,PA).

%% Cartesian product
atom(relation(NR, P, Q, P', Q')):- type_def(NR, NE, NE'),
                                   atom(entity(NE, P, Q)),
                                   atom(entity(NE', P', Q')).


%% Type constraints
2{bad(relation(NR, P, Q, P', Q')); bad(entity(NE, P, Q))} :- atom(relation(NR, P, Q, P', Q')),
											atom(entity(NE, P, Q)),
											type_def(NR, E1, E2),
											E1 != NE.

2{bad(relation(NR, P, Q, P', Q')); bad(entity(NE, P', Q'))} :- atom(relation(NR, P, Q, P', Q')),
											atom(entity(NE, P', Q')),
											type_def(NR, E1, E2),
											E2 != NE.
:- ok(relation(NR, P, Q, P', Q')), ok(entity(NE, P, Q)), type_def(NR, NE', _), NE != NE'.
:- ok(relation(NR, P, Q, P', Q')), ok(entity(NE, P', Q')), type_def(NR, _, NE'), NE != NE'.

%% Bad entity -> bad relation
bad(relation(NR, P, Q, P', Q')) :- atom(relation(NR, P, Q, P', Q')),
					 	 type_def(NR, E, E'),
						 bad(entity(E, P, Q)).

bad(relation(NR, P, Q, P', Q')) :- atom(relation(NR, P, Q, P', Q')),
					 	 type_def(NR, E, E'),
						 bad(entity(E', P', Q')).

%% Relation with itself
bad(relation(NR, P, Q, P, Q)) :- atom(relation(NR, P, Q, P, Q)).
:- ok(relation(_, P, Q, P, Q)).

%% Same span different type
2{bad(entity(NE, P, Q)); bad(entity(NE', P, Q))} :- atom(entity(NE, P, Q)),
									  atom(entity(NE', P, Q)),
									  NE != NE'.
:- ok(entity(NE,P,Q)), ok(entity(NE',P,Q)), NE != NE'.


%% Type inference
ok(entity(E, P, Q)) :- ok(relation(NR, P, Q, P', Q')), type_def(NR, E, E').
ok(entity(E', P', Q')) :- ok(relation(NR, P, Q, P', Q')), type_def(NR, E, E').


%%% Relation exists
relation_flag.
relation_exist :- ok(relation(NR, P, Q, P', Q')).
% :- not relation_exist.

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