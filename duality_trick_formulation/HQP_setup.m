clear
import matlab.unittest.constraints.IsTrue;

yalmip('clear')

n = 10;
no_ineq = 15;
ineq_rank = 5;
no_eq = 15;
eq_rank = 5;

pl = 5;

A_ineq = randn(ineq_rank, n);
A_ineq = [A_ineq; randn(no_ineq - ineq_rank, ineq_rank)*A_ineq];
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

A_eq = randn(eq_rank, n);
A_eq = [A_eq; randn(no_eq - eq_rank, eq_rank)*A_eq];
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

%Use the dualization feature of YALMIP to derive the lexicographic problem
%using the duality trick

F = {};
Fd = {};
obj = {};
objd = {};
w = {};
we = {}
x = sdpvar(n,1);

%Adding the hard constraint
F{1} = [A{1}*x <= b{1}];

%Adding the constraint of the first priority level
w{2} = sdpvar(length(A{2}(:,1)), 1);
F{2} = [F{1}, A{2}*x <= b{2} + w{2}];
%adding equality constraints
we{2} = sdpvar(length(Ae{2}(:,1)), 1);
F{2} = [F{2}, Ae{2}*x - be{2} == we{2}];
obj{2} = norm([w{2}; we{2}], 2);

sum_obj = obj{2}
ops = sdpsettings('solver','mosek', 'verbose', 0);
%test = optimize(F{2}, obj{2}, ops);

for i = 3:pl
    
    %dualize the problem from the previous levle
    [Fd{i-1}, objd{i-1}, primals] = dualize(F{i-1}, obj{i-1});
    w{i} = sdpvar(length(A{i}(:,1)), 1);
    we{i} = sdpvar(length(Ae{i}(:,1)), 1);
    
    F{i} = [F{i-1}, Fd{i-1}]; %Adding the dual feasibility constraints
    F{i} = [F{i}, A{i}*x <= b{i} + w{i}]; %Adding the primal 
    F{i} = [F{i}, Ae{i}*x - be{i} == we{i}];
    %constraints from the next level
    F{i} = [F{i}, obj{i-1} <= objd{i-1}];
    
    obj{i} = norm([w{i}; we{i}] ,2);
    
    sum_obj = sum_obj + obj{i};
    
end

obj_vals = [];
for i = 2:pl
    obj_vals = [obj_vals, obj{i}]; 
end

% dt_sol = optimize(F{pl}, obj{pl}, ops);
% dt_comp_time = dt_sol.solvertime;
% obj_vals = value(obj_vals);



dummy_param = sdpvar(1,1);
P = optimizer(F{pl}, obj{pl}, ops, dummy_param, obj_vals); 
[obj_vals,~,~,~,~,dt_diag] = P(1);
dt_comp_time = dt_diag.solvertime;

%%
%%%%%%%%%%%%
%Now solving the same problem using sequential method

F_seq = {};
x_seq = sdpvar(n,1);
w_seq = {};
w_seq_eq = {};
w_seq_opt = {};
w_seq_eq_opt = {};
obj_seq = {};

for i = 2:pl
    w_seq{i} = sdpvar(length(A{i}(:,1)), 1);
    w_seq_eq{i} = sdpvar(length(Ae{i}(:,1)), 1);
end

%Add the hard constraints
F_seq{2} = [A{1}*x_seq <= b{1}];
%inequality constraints
F_seq{2} = [F_seq{2}, A{2}*x_seq <= b{2} + w_seq{2}];
%equality constraints
F_seq{2} = [F_seq{2}, Ae{2}*x_seq - be{2} == w_seq_eq{2}]; 
obj_seq{2} = norm([w_seq{2}; w_seq_eq{2}], 2)^2;

seq_computation_time = 0;

% sol = optimize(F_seq{2}, obj_seq{2}, ops);
% seq_computation_time = seq_computation_time + sol.solvertime;

P = optimizer(F_seq{2}, obj_seq{2}, ops, dummy_param, [obj_seq{2};w_seq{2};w_seq_eq{2}] ); 
[~,~,~,~,~,dt_diag] = P(1);
seq_computation_time = seq_computation_time + dt_diag.solvertime;

w_seq_opt{2} = value(w_seq{2});
w_seq_eq_opt{2} = value(w_seq_eq{2});
for i = 3:pl
    F_seq{i} = [A{1}*x_seq <= b{1}];
    for j = 2:i-1
        F_seq{i} = [F_seq{i}, A{j}*x_seq <= b{j} + w_seq_opt{j}];
        F_seq{i} = [F_seq{i}, Ae{j}*x_seq - be{j} == w_seq_eq_opt{j}]; 
    end
    
    F_seq{i} = [F_seq{i}, A{i}*x_seq <= b{i} + w_seq{i}];
    F_seq{i} = [F_seq{i}, Ae{i}*x_seq - be{i} == w_seq_eq{i}];
    obj_seq{i} = norm(w_seq{i}, 2)^2 + norm(w_seq_eq{i}, 2)^2;
%     
    sol = optimize(F_seq{i}, obj_seq{i}, ops);
    seq_computation_time = seq_computation_time + sol.solvertime;
    
%     P = optimizer(F_seq{i}, obj_seq{i}, ops, dummy_param, obj_seq{i}); 
%     [~,~,~,~,~,dt_diag] = P(1);
%     seq_computation_time = seq_computation_time + dt_diag.solvertime;

    w_seq_opt{i} = value(w_seq{i});
    w_seq_eq_opt{i} = value(w_seq_eq{i});
    
end

%Check if the result from both the methods are identical
allGood = true;

obj_vals_seq = [];
for i = 2:pl
    obj_vals_seq = [obj_vals_seq, sqrt(value(obj_seq{i}))];
    allGood = allGood*(abs(obj_vals(i-1) - obj_vals_seq(i-1))/obj_vals(i-1) <= 1e-4 || abs(obj_vals(i-1) - obj_vals_seq(i-1)) <= 1e-4);
    
    
end

allGood
obj_vals
obj_vals_seq

