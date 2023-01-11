% projection

atom(A) :- atom(A,P).
prob(A,P) :- atom(A,P).

prod(P, A) :- atom(A, P), ok(A).
invprod(P, A) :- atom(A, P), nok(A).


%% overlapping

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