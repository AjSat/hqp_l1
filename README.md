# Repository for weighted method for strictly hierarchical linear programming with applications to robot control

### Using L1 norm as a regularization step can allow for sparse or parsimonious control of the robot system.
### Specifying constraint violations using L1 norm leads to sparse constraint violations
## instructions on running the code to reproduce results coming by Jan 26.

This repository contains the source code for a paper submitted to IEEE RA-L/ICRA and is currently under review.
Documentation coming soon to improve readability of the code.


## Duality trick reformulation:

The source code for duality trick reformulation of the lexicographic linear program into a single objective linear program can be found in the duality trick folder. To run this file, MATLAB and Yalmip toolbox needs to be installed. To reproduce the computational performance reported in the paper, it is also recommended to install Gurobi which provides a free academic license.

There is also an undocumented implementation of the duality trick also for hierarchical quadratic programming.

https://yalmip.github.io/

https://www.gurobi.com/downloads/gurobi-optimizer-eula/

## Weighted and sequential methods 

The code for this part includes the following dependencies:

* **Python3.5** or above

* **CasADi** - A toolbox used for automatic differentiation of the task functions and as the interface to optimization solvers. Can be installed by
```
pip3 install casadi
```
* **PyBullet** - If one chooses to simulate and visualize the robot motion. Can be installed by:

```
pip3 install pybullet
```


The weighted and sequential problems with L1 norm penalty are implemented in Python using Casadi in the **hqp.py** file. Within this file:

* hqp.solve_cascadedQP4() - Is the implementation of the sequential method.
* solve_HQPl1() is the implementation of the weighted method.
* solve_adaptive_hqp3() is a newly implemented adaptive method whose verification of lexicographic optimality is exact because simply the corresponding sequential method is used for verification.



