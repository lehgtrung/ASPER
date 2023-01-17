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
%2{bad(entity(CH, C1, C2)); bad(entity(PA, P1, P2))}: :- atom(entity(CH, C1, C2)),
%                                                        atom(entity(PA, P1, P2)),
%                                                        C1 >= P1, C2 <= P2,
%                                                        PA != CH,
%                                                        PA != task,
%                                                        PA != otherScientificTerm,
%                                                        PA != generic,
%                                                        CH != otherScientificTerm,
%                                                        CH != generic.

%ok(entity(CH, C1, C2)); ok(entity(PA, P1, P2)), atom(entity(CH, C1, C2)),
%                                                atom(entity(PA, P1, P2)),
%                                                C1 >= P1, C2 <= P2,
%                                                PA != CH,
%                                                PA != task,
%                                                PA != otherScientificTerm,
%                                                PA != generic,
%                                                CH != otherScientificTerm,
%                                                CH != generic.


%For Compare and Conjunction: compare abbreviation with original concept, usually abbreviation is written first (cond1),
% sometimes written latter but in generic form (those, that,...)
% and the head (abbreviation) is shorter than the tail (original concept)
