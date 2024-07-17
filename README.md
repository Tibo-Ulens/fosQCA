# Family of Sets QCA

A modification of the standard QCA algorithm that can take multiple sets as arguments

## Algorithm Outline

1. Generate and merge all possible rules for each input set
1. Select all possible subsets of merged rules that meet the consistency and
   coverage thresholds for all input sets
1. Sort by highest average consistency and coverage
