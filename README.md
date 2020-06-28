# hqp_l1

A illustrative toolbox for hqp_l1.

Structure:

Create variable. {weight. - Quadratic penalty for regularization.}

Create constraint{
	Weight. 
	Priority. 0 - highest. 1 - 2nd highest and so on. -1 only soft constraint with quadratic penalty.
	Equality. 
	Less than. 
	Greater than. 
	LUB.
} 

Functions:
Compute gains - computes the gains on the L1 norm for all slack variables.
Check hierarchy - checks if the hierarchy is satisfied for all variables.
Update gains - updates gains only for a certain  hierarchy and upwards if hierarchy not satisfied.
run L1-HQP
run cascaded-HQP

