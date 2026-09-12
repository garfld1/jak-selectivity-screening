# Final shortlist

273 patterns passed all pre-specified filters. Recurrence is ranked separately from any single pattern.

## Feature/residue recurrence

- J1:LEU959:hydrogen_bond: 163
- J1:ALA906:hydrophobic: 127
- J1:ALA906: 127
- J1:ARG1007: 103
- J1:LEU881:hydrophobic: 89
- J1:LEU959: 79
- J1:VAL889:hydrophobic: 74
- J1:VAL889: 70
- J1:LEU881: 66
- J2:LEU855:hydrophobic: 64
- J1:LEU1010: 63
- J1:LEU1010:hydrophobic: 63
- J2:LEU855: 47
- J1:ARG1007:hydrophobic: 42
- J1:ARG879: 20
- J2:TYR931: 17
- J2:VAL863:hydrophobic: 8
- J2:VAL863: 8
- J1:ARG879:hydrogen_bond: 4
- J2:TYR931:hydrophobic: 2

## Candidate pharmacophore regions

- `J1:LEU959:hydrogen_bond`: present in 163 survivors; median validation enrichment 1.36; median specificity 2.94.
- `J1:ALA906:hydrophobic`: present in 127 survivors; median validation enrichment 1.36; median specificity 2.91.
- `J1:ALA906`: present in 127 survivors; median validation enrichment 1.36; median specificity 2.91.
- `J1:ARG1007`: present in 103 survivors; median validation enrichment 1.58; median specificity 2.47.

Bare residue features: 10; typed interaction features: 10.

## Interpretation

PPV and enrichment are reported against each stage's own baseline in the CSV; magnitudes are not directly comparable across discovery, random holdout, scaffold holdout, and Ki validation. Ki validation also differs from IC50 discovery. A feature that also appears among JAK1-search hits is a generic-binder red flag and must be cross-checked against `specificity_score`, not treated as independent support.
