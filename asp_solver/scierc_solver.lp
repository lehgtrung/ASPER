% projection

atom(A) :- atom(A,P).
prob(A,P) :- atom(A,P).

prod(P, A) :- atom(A, P), ok(A).
invprod(P, A) :- atom(A, P), nok(A).

%% Notes:
%% - Entities: Task, Method, Material, OtherScientificTerm, Metric, Generic
%% - Relations: Used-for, Feature-of, Hyponym-of, Evaluate-for, Part-of, Compare(symmetric), Conjunction(symmetric)

%% [Rule]: a subword must be completely inside a word or completely outside of it
% Do this with every combination of entity types
2{bad(task(P1,P2)), bad(task(Q1,Q2))} :- atom(task(P1,P2)), atom(task(Q1,Q2)), P1 < Q1, P2 >= Q1.



%% [Rule]: Compare, Conjunction, Part-of need to have the same entity type or at
%% least 1 entity is of type 'Generic' or 'OtherScientificTerm'.

% compare
bad(compare(P1,P2,Q1,Q2)) :- task(P1,P2), method(Q1,Q2).
bad(compare(P1,P2,Q1,Q2)) :- method(P1,P2), task(Q1,Q2).

bad(compare(P1,P2,Q1,Q2)) :- task(P1,P2), material(Q1,Q2).
bad(compare(P1,P2,Q1,Q2)) :- material(P1,P2), task(Q1,Q2).

bad(compare(P1,P2,Q1,Q2)) :- task(P1,P2), metric(Q1,Q2).
bad(compare(P1,P2,Q1,Q2)) :- metric(P1,P2), task(Q1,Q2).

% conjunction
bad(conjunction(P1,P2,Q1,Q2)) :- task(P1,P2), method(Q1,Q2).
bad(conjunction(P1,P2,Q1,Q2)) :- method(P1,P2), task(Q1,Q2).

bad(conjunction(P1,P2,Q1,Q2)) :- task(P1,P2), material(Q1,Q2).
bad(conjunction(P1,P2,Q1,Q2)) :- material(P1,P2), task(Q1,Q2).

bad(conjunction(P1,P2,Q1,Q2)) :- task(P1,P2), metric(Q1,Q2).
bad(conjunction(P1,P2,Q1,Q2)) :- metric(P1,P2), task(Q1,Q2).

% partOf
bad(partOf(P1,P2,Q1,Q2)) :- task(P1,P2), method(Q1,Q2).
bad(partOf(P1,P2,Q1,Q2)) :- method(P1,P2), task(Q1,Q2).

bad(partOf(P1,P2,Q1,Q2)) :- task(P1,P2), material(Q1,Q2).
bad(partOf(P1,P2,Q1,Q2)) :- material(P1,P2), task(Q1,Q2).

bad(partOf(P1,P2,Q1,Q2)) :- task(P1,P2), metric(Q1,Q2).
bad(partOf(P1,P2,Q1,Q2)) :- metric(P1,P2), task(Q1,Q2).


%For Compare and Conjunction: compare abbreviation with original concept, usually abbreviation is written first (cond1),
% sometimes written latter but in generic form (those, that,...)
% and the head (abbreviation) is shorter than the tail (original concept)

3{bad(compare(P1,P2,Q1,Q2)), bad(type(P1,P2)), bad(type(P1,P2))}1:- atom(compare(P1,P2,Q1,Q2)), P1 > Q1, P2-P1 < Q2-Q1,
not atom(generic(P1,P2)), type(P1,P2) != type(Q1,Q2).