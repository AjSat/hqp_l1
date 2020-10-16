clear

yalmip('clear')

n = 20; %Number of decision variables in the system
no_ineq = 20; %Number of inequality constraints
ineq_rank = 10; %The rank of the inequality constraints
no_eq = 15; %Number of equality constraints
eq_rank = 5; %The rank of the equality constraints

pl = 3; %The number of priority levels (including the hard level)
debug_mode = true;

%Randomly generating inequality constraints
A_ineq = randn(ineq_rank, n);
A_ineq = [A_ineq; randn(no_ineq - ineq_rank, ineq_rank)*A_ineq];
A_ineq = A_ineq(randperm(size(A_ineq, 1)), :); %Random shuffling
A_ineq = normr(A_ineq);
b_ineq = randn(no_ineq, 1);
A = {};
b = {};
ineq_con_per_level = floor(no_ineq / pl);
for i = 1:pl-1
    A{i} = A_ineq((i-1)*ineq_con_per_level + 1:(i)*ineq_con_per_level, :);
    b{i} = b_ineq((i-1)*ineq_con_per_level + 1:(i)*ineq_con_per_level);
end
A{pl} = A_ineq((i)*ineq_con_per_level+1:end,:);
b{pl} = b_ineq((i)*ineq_con_per_level+1:end,:);

%Randomly generating equality constraints
A_eq = randn(eq_rank, n);
A_eq = [A_eq; randn(no_eq - eq_rank, eq_rank)*A_eq];
A_eq = A_eq(randperm(size(A_eq, 1)), :); %Random shuffling
A_eq = normr(A_eq);
b_eq = randn(no_eq, 1);
Ae = {};
be = {};
eq_con_per_level = floor(no_eq / (pl-1));
for i = 1:pl-2
    Ae{i+1} = A_eq((i-1)*eq_con_per_level + 1:(i)*eq_con_per_level, :);
    be{i+1} = b_eq((i-1)*eq_con_per_level + 1:(i)*eq_con_per_level);
end
Ae{pl} = A_eq((pl-2)*eq_con_per_level+1:end,:);
be{pl} = b_eq((pl-2)*eq_con_per_level+1:end,:);

%Using the dualization feature of YALMIP, the lexicographic problem
%is formulated as a single LP.

F = {}; %Primal feasibility constraints upto a priority level
Fd = {}; %Dual feasibility constraints upto a priority level
obj = {}; %Objective functions of the primal problem upto a priority level
objd = {}; %Dual objective functions upto a priority level
w = {}; %slack variables for the inequality constraints of a priority level
we = {}; %slack variables for the equality constraints of a priority level

x = sdpvar(n,1); %Decision variables of the problem

%Adding the hard constraints of the highest priority level
F{1} = [A{1}*x <= b{1}];

%Adding the constraint of the second (the first relaxable level) priority level
w{2} = sdpvar(length(A{2}(:,1)), 1); %creating the corresponding slack variables
F{2} = [F{1}, A{2}*x <= b{2} + w{2}, w{2} >= 0];
%adding equality constraints
we{2} = sdpvar(length(Ae{2}(:,1)), 1);
F{2} = [F{2}, -we{2} <= Ae{2}*x - be{2} <= we{2}];

obj{2} = sum(w{2}) + sum(we{2}); %Adding L1 penalties on the constraint 
%violations of the second level

sum_obj = obj{2};
if debug_mode
    ops = sdpsettings('solver','gurobi', 'verbose', 1);
    dt_sol = optimize(F{2}, obj{2}, ops);
    %strange error observed in debbug mode for 3 levels. The duality trick
    %constraint is not taken into account by Yalmip and is violated by the
    %solution
else
    ops = sdpsettings('solver','gurobi', 'verbose', 0);
end

for i = 3:pl
    
    %dualize the problem from the previous level
    [Fd{i-1}, objd{i-1}, primals] = dualize(F{i-1}, obj{i-1});
    
    %creating slack variables for this level
    w{i} = sdpvar(length(A{i}(:,1)), 1);
    we{i} = sdpvar(length(Ae{i}(:,1)), 1);
    
    %Adding the dual feasibility constraints of the previous level
    F{i} = [F{i-1}, Fd{i-1}]; 
    F{i} = [F{i}, obj{i-1} <= objd{i-1}]; %duality trick
    %Adding the primalconstraints from the next level
    F{i} = [F{i}, w{i} >=0, A{i}*x <= b{i} + w{i}];  %inequality constraints
    F{i} = [F{i}, -we{i} <= Ae{i}*x - be{i} <= we{i}]; %equality constraints
    
    
    obj{i} = sum(w{i}) + sum(we{i}); %Formulating the objective of this level
    
    sum_obj = sum_obj + obj{i};
    
end

%Run the duality trick formulation for this problem
if debug_mode
    F{3} = [F{3}, obj{2} <= objd{2}]; %duality trick
    dt_sol = optimize(F{pl}, obj{pl}, ops);
    dt_comp_time = dt_sol.solvertime;
    objp2 = value(obj{2});
    objd2 = value(objd{2});
else
    dummy_param = sdpvar(1,1);
    P = optimizer(F{pl}, obj{pl}, ops, dummy_param, sum_obj);
    [~,~,~,~,~,dt_diag] = P(1);
    dt_comp_time = dt_diag.solvertime;
end

%%%%%%%%%%%%
%Now solving the same problem using sequential method

F_seq = {}; %primal problems of the sequential method
x_seq = sdpvar(n,1); %the decision variables
w_seq = {}; %the slack variables for inequality constraints 
w_seq_eq = {}; %the slack variables for equality constraints
obj_seq = {}; %objective functions for each level of the method

%instantiating the slack variables for each level
for i = 2:pl
    w_seq{i} = sdpvar(length(A{i}(:,1)), 1);
    w_seq_eq{i} = sdpvar(length(Ae{i}(:,1)), 1);
end

%Add the hard constraints
F_seq{2} = [A{1}*x_seq <= b{1}];
%inequality constraints
F_seq{2} = [F_seq{2}, A{2}*x_seq <= b{2} + w_seq{2}, w_seq{2} >= 0];
%equality constraints
F_seq{2} = [F_seq{2}, -w_seq_eq{2} <= Ae{2}*x_seq - be{2} <= w_seq_eq{2}]; 
obj_seq{2} = sum(w_seq{2}) + sum(w_seq_eq{2});

%solve the sequential method for the first priority level
seq_computation_time = 0;
if debug_mode
    sol = optimize(F_seq{2}, obj_seq{2}, ops);
    seq_computation_time = seq_computation_time + sol.solvertime;
else
    P = optimizer(F_seq{2}, obj_seq{2}, ops, dummy_param, obj_seq{2});
    [~,~,~,~,~,dt_diag] = P(1);
    seq_computation_time = seq_computation_time + dt_diag.solvertime;
end

for i = 3:pl
    F_seq{i} = F_seq{i-1}; %retain all the primal feasibility constraints from previous level  
    
    %compute the optimal value of the slack variables from previous step of
    %the sequential method
    w_ineq_opt = value(w_seq{i-1}); 
    w_eq_opt = value(w_seq_eq{i-1});
    
    %add the constraints from the ith priority level
    F_seq{i} = [F_seq{i}, sum(w_seq{i-1}) + sum(w_seq_eq{i-1}) <= sum(w_ineq_opt) + sum(w_eq_opt)];
    F_seq{i} = [F_seq{i}, A{i}*x_seq <= b{i} + w_seq{i}, w_seq{i} >= 0];
    F_seq{i} = [F_seq{i}, -w_seq_eq{i} <= Ae{i}*x_seq - be{i} <= w_seq_eq{i}];
    
    %define the objective for the ith level of the method
    obj_seq{i} = sum(w_seq{i}) + sum(w_seq_eq{i});
    
    if debug_mode
        sol = optimize(F_seq{i}, obj_seq{i}, ops);
        seq_computation_time = seq_computation_time + sol.solvertime;
    else
        P = optimizer(F_seq{i}, obj_seq{i}, ops, dummy_param, obj_seq{i});
        [~,~,~,~,~,dt_diag] = P(1);
        seq_computation_time = seq_computation_time + dt_diag.solvertime;
    end
    
end

%Check if the result from both the methods are identical
allGood = true;
obj_vals = [];
for i = 2:pl
    allGood = allGood*(abs(value(obj{i}) - value(obj_seq{i})) <= 1e-6);
    obj_vals = [obj_vals, value(obj{i})];
    
end

allGood
obj_vals

