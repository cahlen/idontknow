Ramsey R(5,5) SAT Encodings

ramsey_k43_naive.cnf:
  903 vars, 1,925,237 clauses. Basic encoding + minimal symmetry.
  Result: INTRACTABLE (98 solvers × 2 hours, 0% var elimination)

ramsey_k43_degree_constrained.cnf:
  87,591 vars (903 edge + 86,688 aux), 2,098,097 clauses.
  Degree constraints from R(4,5)=25: 18 ≤ red_deg(v) ≤ 24.
  Result: 88% variable elimination in preprocessing.
