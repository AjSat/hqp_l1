# Repository with source code for IEEE RA-L/ICRA submission:

This repository contains the source code for a paper submitted to IEEE RA-L/ICRA and is currently under review.


Code unreadable at the moment.

Refactoring and documentation coming soon!

## A minor error in equation 11 that does not affect the rest of the paper

There is minor error in equation 11 that does not affect the heuristic of the adaptive method nor the conclusions of the paper. The correction for this can be found below.

[Correction](https://www.dropbox.com/s/prtiq1yv7mf0oo7/error.pdf?dl=0)

## Duality trick reformulation:

The source code for duality trick reformulation of the lexicographic linear program into a single objective linear program can be found in the duality trick folder. To run this file, MATLAB and Yalmip toolbox needs to be installed. To reproduce the computational performance reported in the paper, it is also recommended to install Gurobi which provides a free academic license.

There is also an undocumented implementation of the duality trick also for hierarchical quadratic programming.

https://yalmip.github.io/

https://www.gurobi.com/downloads/gurobi-optimizer-eula/
