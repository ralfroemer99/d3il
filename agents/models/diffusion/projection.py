import numpy as np
import torch
import time
from scipy.optimize import minimize, Bounds

class Projector:

    def __init__(self, horizon, transition_dim, constraint_list=[], scaler=None, 
                 dt=0.1, skip_initial=True, gradient_weights=None, device='cuda'):
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.dt = dt
        self.skip_initial = skip_initial
        self.gradient_weights = gradient_weights
        self.device = device
        
        # Determine whether to include actions in the projection
        if scaler is None:
            self.a_mean = np.zeros(transition_dim)
            self.A_std = np.ones(transition_dim)
        else:
            self.a_mean = scaler.y_mean.cpu().numpy()
            self.a_std = scaler.y_std.cpu().numpy()

        # Quadratic cost
        # self.Q = np.eye(transition_dim * horizon).astype('double')
        self.Q = np.kron(np.eye(horizon), np.diag(self.a_std ** 2)).astype('double')
        self.A = np.empty((0, transition_dim * horizon)).astype('double')        # Equality constraints
        self.b = np.empty(0).astype('double')
        self.C = np.empty((0, transition_dim * horizon)).astype('double')        # Inequality constraints
        self.d = np.empty(0).astype('double')

        self.halfspace_constraints = HalfspaceConstraints(horizon=horizon, transition_dim=transition_dim, a_mean=self.a_mean, a_std=self.a_std, skip_initial=self.skip_initial, device=self.device)
        self.dynamic_constraints = DynamicConstraints(horizon=horizon, transition_dim=transition_dim, dt=dt, a_mean=self.a_mean, a_std=self.a_std, skip_initial=self.skip_initial, device=self.device)
        self.ellipsoidal_constraints = EllipsoidalConstraints(horizon=horizon, transition_dim=transition_dim, a_mean=self.a_mean, a_std=self.a_std, device=self.device)

        for constraint_spec in constraint_list:
            if constraint_spec[0] == 'deriv':
                self.dynamic_constraints.constraint_list.append(constraint_spec)
            elif constraint_spec[0] == 'lb' or constraint_spec[0] == 'ub' or constraint_spec[0] == 'eq' or constraint_spec[0] == 'ineq':
                self.halfspace_constraints.constraint_list.append(constraint_spec)
            elif constraint_spec[0] == 'sphere_inside' or constraint_spec[0] == 'sphere_outside':
                self.ellipsoidal_constraints.constraint_list.append(constraint_spec)

        self.halfspace_constraints.build_matrices()
        self.dynamic_constraints.build_matrices()
        self.ellipsoidal_constraints.build_matrices()
        self.append_linear_constraint(self.halfspace_constraints)
        self.append_linear_constraint(self.dynamic_constraints)
        # self.add_numpy_constraints()     

    def project(self, trajectory, state, constraint_info=None):
        """
            trajectory: np.ndarray of shape (batch_size, horizon, transition_dim)
            Solve an optimization problem of the form 
                \hat z =   argmin_z 1/2 z^T Q z + r^T z
                        subject to  Az  = b
                                    Cz <= d
            where z = (o_0, o_1, ..., o_{H-1}) is the trajectory in vector form. The matrices A, b, C, and d are defined by the dynamic and safety constraints.
                                    
        """
        
        trajectory = trajectory.cpu().numpy().astype('double')

        batch_size, horizon, transition_dim = trajectory.shape

        # Reshape the trajectory to a batch of vectors (from B x H x T to B x (HT)
        trajectory = trajectory.reshape(trajectory.shape[0], -1)

        # Cost
        r = - trajectory @ self.Q
        Q = self.Q

        # Constraints
        A = self.A
        b = self.b
        C = self.C
        d = self.d

        if self.skip_initial:
            s_0 = trajectory[0, :self.transition_dim]
            counter = 0
            for constraint in self.dynamic_constraints.constraint_list:
                if constraint[0] == 'deriv':
                    x_idx = int(constraint[1][0])
                    b[counter * self.horizon] = s_0[x_idx]
                    counter += 1

        # Constraints
        nlp_constraints = ()
        for constraint_idx in range(len(self.ellipsoidal_constraints.P_list)):
            P = self.ellipsoidal_constraints.P_list[constraint_idx]
            q = self.ellipsoidal_constraints.q_list[constraint_idx]
            v = self.ellipsoidal_constraints.v_list[constraint_idx]
            for t in range(1, self.horizon):                        # Obstacle constraints
                start_idx = t * self.transition_dim
                end_idx = (t + 1) * self.transition_dim
                nlp_constraints += ({'type': 'ineq', 'fun': lambda x, start_idx=start_idx, end_idx=end_idx, P=P, q=q, v=v: -x[start_idx: end_idx] @ P @ x[start_idx: end_idx] - q @ x[start_idx: end_idx] + v,
                                    'jac': lambda x, start_idx=start_idx, end_idx=end_idx, P=P, q=q: np.concatenate([np.zeros(start_idx), -2 * P @ x[start_idx: end_idx] - q, np.zeros(len(x) - end_idx)])},)

        if C.size > 0:
            nlp_constraints += ({'type': 'ineq', 'fun': lambda x: -C @ x + d, 'jac': lambda x: -C},)
        if A.size > 0:
            nlp_constraints += ({'type': 'eq', 'fun': lambda x: A @ x - b, 'jac': lambda x: A},)   
        
        projection_costs = np.ones(batch_size, dtype=np.float32)
        sol_np = np.zeros((batch_size, self.horizon * self.transition_dim), dtype=np.float32)
        for i in range(batch_size):
            # Cost
            cost_fun = lambda x: 0.5 * x @ Q @ x + r[i] @ x # + (A_double @ x - b_double) @ (A_double @ x - b_double)
            jac_cost_fun = lambda x: Q @ x + r[i]
            res = minimize(fun=cost_fun, 
                            x0=trajectory[i],
                            constraints=nlp_constraints, 
                            method='SLSQP', 
                            jac=jac_cost_fun, 
                            bounds=Bounds(-5 * np.ones_like(trajectory[i]), 5 * np.ones_like(trajectory[i])),
                            tol=1e-6,
                            options={'maxiter': 1000, 'disp': False})
            sol_np[i] = res.x
            projection_costs[i] = 0.5 * sol_np[i] @ Q @ sol_np[i] + r[i] @ sol_np[i] + 0.5 * trajectory[i] @ Q @ trajectory[i]

            # if np.linalg.norm(A_double @ res.x - b_double) > 1e-3:
            #     print('Equality constraints not satisfied!')
            # if np.any(C_double @ res.x > d_double + 1e-3):
            #     print('Inequality constraints not satisfied!')

        sol = torch.tensor(sol_np, device=self.device).reshape((batch_size, horizon, transition_dim)).float()

        # print(f'Projection time {self.solver}:', time.time() - start_time)
        return sol, projection_costs    # only implemented for proxsuite and scipy and parallelize=False
    
    def compute_gradient(self, trajectory, constraints=None):
        """
            trajectory: np.ndarray of shape (batch_size, horizon, transition_dim) or (horizon, transition_dim)
            Calculate the (weighted) gradients for the following cost functions:
            c_1 = ||A * tau - b||^2                             --> grad_1 = 2 * A^T (A * tau - b)
            c_2 = max(0, C * tau - d)^2                         --> grad_2 = 2 * C^T max(0, C * tau - d)
            c_3 = sum_{t=1}^{H-1} (s_t^T P s_t + q^T s_t - v)^2 --> grad_3 = 2 * (2 * P s_t + q) * (s_t^T P s_t + q^T s_t - v)                 
        """
        
        trajectory_reshaped = trajectory.reshape(trajectory.shape[0], -1)
        trajectory_np = trajectory_reshaped.cpu().numpy()

        # Constraints
        A, b, C, d = self.A, self.b, self.C, self.d

        if self.skip_initial:
            s_0 = trajectory_reshaped[0, :self.transition_dim]
            counter = 0
            for constraint in self.dynamic_constraints.constraint_list:
                if constraint[0] == 'deriv':
                    x_idx = int(constraint[1][0])
                    b[counter * self.horizon] = s_0[x_idx]
                    counter += 1

        # Equality and polytopic constraints
        grad1 = torch.zeros_like(trajectory_reshaped)
        grad2 = torch.zeros_like(trajectory_reshaped)
        for i in range(trajectory.shape[0]):
            grad1[i] = - A.T @ (A @ trajectory_reshaped[i] - b)
            grad2[i] = - C.T @ torch.max(torch.zeros_like(C @ trajectory_reshaped[i] - d), C @ trajectory_reshaped[i] - d)
        grad1 = grad1.reshape(trajectory.shape)
        grad2 = grad2.reshape(trajectory.shape)

        # Obstacle constraints
        grad3 = np.zeros_like(trajectory_np)
        for constraint_idx in range(len(self.ellipsoidal_constraints.P_list)):
            P = self.ellipsoidal_constraints.P_list[constraint_idx]
            q = self.ellipsoidal_constraints.q_list[constraint_idx]
            v = self.ellipsoidal_constraints.v_list[constraint_idx]
            for t in range(1, self.horizon):
                start_idx = t * self.transition_dim
                end_idx = (t + 1) * self.transition_dim
                for i in range(trajectory.shape[0]):
                    if trajectory_np[i, start_idx: end_idx] @ P @ trajectory_np[i, start_idx: end_idx] + q @ trajectory_np[i, start_idx: end_idx] <= v:
                        continue
                    else:
                        grad3[i, start_idx: end_idx] -= 2 * P @ trajectory_np[i, start_idx: end_idx] + q   
        grad3 = torch.tensor(grad3, device=self.device).reshape(trajectory.shape)
        
        if self.gradient_weights is not None:
            grad1 = grad1 * self.gradient_weights[0]
            grad2 = grad2 * self.gradient_weights[1]
            grad3 = grad3 * self.gradient_weights[2]
        
        return grad1 + grad2 + grad3 

    def append_linear_constraint(self, constraint):
        self.C = np.concatenate((self.C, constraint.C), axis=0)
        self.d = np.concatenate((self.d, constraint.d), axis=0)
        self.A = np.concatenate((self.A, constraint.A), axis=0)
        self.b = np.concatenate((self.b, constraint.b), axis=0)
        if constraint.__class__.__name__ == 'HalfspaceConstraints':
            self.A_safe, self.b_safe, self.C_safe, self.d_safe = constraint.A, constraint.b, constraint.C, constraint.d
        elif constraint.__class__.__name__ == 'DynamicConstraints':
            self.A_dyn, self.b_dyn, self.C_dyn, self.d_dyn = constraint.A, constraint.b, constraint.C, constraint.d

    # def add_numpy_constraints(self):
    #     self.A_np = self.A.cpu().numpy().astype('double')
    #     self.b_np = self.b.cpu().numpy().astype('double')     
    #     self.C_np = self.C.cpu().numpy().astype('double')
    #     self.d_np = self.d.cpu().numpy().astype('double')
    #     self.Q_np = self.Q.cpu().numpy().astype('double')


class Constraints:

    def __init__(self, horizon, transition_dim, a_mean, a_std, device='cuda'):
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.a_mean = a_mean
        self.a_std = a_std
        self.device = device

        self.constraint_list = []

        self.A = np.empty((0, self.transition_dim * self.horizon)).astype('double')
        self.b = np.empty(0).astype('double')
        self.C = np.empty((0, self.transition_dim * self.horizon)).astype('double')
        self.d = np.empty(0).astype('double')

    def build_matrices(self):
        pass

class HalfspaceConstraints(Constraints):

    def __init__(self, skip_initial=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_initial = skip_initial
        
    def build_matrices(self, constraint_list=None):
        """
            Input:
                constraint_list: list of constraints
                    e.g. [('lb', [-1.0, -inf, 0]), ('ub', [1.0, 2.0, inf]), ('eq', ([0, 1, 1], 1.5)), ('ineq': ([1, 0, 0], 0.5))] -->
                    x_0 in [-1, 1], x_1 in [-inf, 2], x_2 in [0, inf], 0 * x_0 + 1 * x_1 + 1 * x_2 = 1.5, 1 * x_0 + 0 * x_1 + 0 * x_2 <= 0.5,
                    where x_i is the i-th dimension of the state or state-action vectors
            The matrices have the following shapes:
                C: (horizon * n_bounds, transition_dim * horizon)
                d: (horizon * n_bounds)
                lb: horizon * transition_dim
            C consists of n_bounds blocks of shape (horizon, transition_dim * horizon), where each block corresponds to a 
            constraint (ub or lb) on a specific dimension. The block has a 1 or -1 at the corresponding dimension and time step.
        """

        if constraint_list is None:
            constraint_list = self.constraint_list
        else:
            self.constraint_list.extend(constraint_list)

        for constraint in constraint_list:
            type = constraint[0]
            bound = constraint[1]
            if type == 'lb' or type == 'ub':
                for dim in range(len(bound)):
                    if bound[dim] == -np.inf or bound[dim] == np.inf:
                        continue
                    
                    mat_append = np.zeros((self.horizon, self.transition_dim * self.horizon))
                    vec_append = np.zeros(self.horizon)

                    sign = 1 if type == 'ub' else -1
                    for t in range(self.horizon):
                        mat_append[t, t * self.transition_dim + dim] = sign
                        vec_append[t] = sign * bound[dim]
                        
                    if self.scaler is not None:
                        x_min = self.scaler.mins[dim]
                        x_max = self.scaler.maxs[dim]
                        mat_append = mat_append * (x_max - x_min) / 2
                        vec_append = vec_append - sign * (x_min + x_max) / 2

                    if self.skip_initial and dim >= self.action_dim:
                        mat_append = mat_append[1:]
                        vec_append = vec_append[1:]

                    self.C = np.concatenate((self.C, mat_append), axis=0)
                    self.d = np.concatenate((self.d, vec_append), axis=0)
                continue         

            # type == 'eq' or 'ineq'
            mat_append = np.zeros((self.horizon, self.transition_dim * self.horizon))
            vec_append = np.zeros(self.horizon)

            for i in range(self.horizon):
                if self.scaler is not None:
                    x_min = self.scaler.mins
                    x_max = self.scaler.maxs

                    # Unnormalize the constraints. TODO: Extend to actions by checking if dim <= action_dim
                    # We have Cs <= d, where s is the unnormalized state vector. This is converted to C's_n <= d', where s_n is the normalized state vector,
                    # by using the fact that s = (s_n + 1) * (s_max - s_min) / 2 + s_min.
                    a = bound[0] * (x_max - x_min) / 2
                    b = bound[1] - bound[0] @ (x_max + x_min) / 2
                else:
                    a = bound[0]
                    b = bound[1]
                
                mat_append[i, i * self.transition_dim: (i + 1) * self.transition_dim] = a
                vec_append[i] = b

            if self.skip_initial:
                mat_append = mat_append[1:]
                vec_append = vec_append[1:]

            if type == 'eq':
                self.A = np.concatenate((self.A, mat_append), axis=0)
                self.b = np.concatenate((self.b, vec_append), axis=0)
            else:
                self.C = np.concatenate((self.C, mat_append), axis=0)
                self.d = np.concatenate((self.d, vec_append), axis=0)       

class DynamicConstraints(Constraints):
    def __init__(self, dt=0.02, skip_initial=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dt = dt
        self.skip_initial = skip_initial
        
    def build_matrices(self, constraint_list=None):
        """
            Input:
                constraint_list: list of constraints
                    e.g. [('deriv', [0, 2]), ('deriv', [1, 3])] -->
                    x_0[t+1] = x_0[t] + self.dt * x_2[t], x_1[t+1] = x_1[t] + self.dt * x_3[t]                                      (explicit Euler) or
                    x_0[t+1] = x_0[t] + self.dt * (x_2[t] + x_2[t+1]) / 2, x_1[t+1] = x_1[t] + self.dt * (x_3[t] + x_3[t+1]) / 2    (variant of trapezoidal rule)
                    where x_i[t] is the i-th dimension of the state or state-action vectors at time t
            The matrices have the following shapes:
                C: (horizon * n_bounds, transition_dim * horizon)
                d: (horizon * n_bounds)
        """

        if constraint_list is None:
            constraint_list = self.constraint_list
        else:
            self.constraint_list.extend(constraint_list)
        
        for constraint in constraint_list:
            type = constraint[0]
            dims = constraint[1]
            if 'deriv' in type:
                x_idx = int(dims[0])
                dx_idx = int(dims[1])
            
                mat_append = np.zeros((self.horizon - 1, self.transition_dim * self.horizon))
                vec_append = np.zeros(self.horizon - 1)

                # Calculate multiplicative factors needed for normalization
                dx_mean = self.a_mean[dx_idx]
                x_std = self.a_std[x_idx]
                dx_std = self.a_std[dx_idx]

                for i in range(self.horizon - 1):
                    mat_append[i, i * self.transition_dim + x_idx] = x_std
                    mat_append[i, i * self.transition_dim + dx_idx] = self.dt * dx_std
                    mat_append[i, (i + 1) * self.transition_dim + x_idx] = -x_std
                    vec_append[i] = -dx_mean * self.dt

                # if self.skip_initial:     # --> Do that in the projection method because it needs the current state. For that, record the relevant rows
                #     mat_fix_initial = torch.zeros(1, self.transition_dim * self.horizon, device=self.device)    # Fix the initial state
                #     mat_fix_initial[0, x_idx] = 1
                #     mat_append = torch.cat((mat_fix_initial, mat_append), dim=0)
                #     vec_append = torch.cat((torch.tensor([0], device=self.device), vec_append), dim=0)          # Must be changed to current state in each iteration!

                self.A = np.concatenate((self.A, mat_append), axis=0)
                self.b = np.concatenate((self.b, vec_append), axis=0)

class EllipsoidalConstraints(Constraints):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_matrices(self, constraint_list=None):
        """
            Input:
                constraint_list: list of constraints
                    e.g. [('sphere_inside', [0, 2], [-1, 5], 1), ('sphere_outside', [1, 3], [0, 1], 4)] -->
                    (x_0 + 1)^2 + (x_2 - 5)^2 <= 1, x_1^2 + (x_3 - 1)^2 >= 4
                    where x_i is the i-th dimension of the state or state-action vectors at time t
            Generate the matrix P and the vector q for:
                s_i^T P s_i + q^T s_i <= v
            The matrices have the following shapes:
                C: (horizon * n_bounds, transition_dim * horizon)
                d: (horizon * n_bounds)
        """

        if constraint_list is None:
            constraint_list = self.constraint_list
        else:
            self.constraint_list.extend(constraint_list)

        self.P_list = []
        self.q_list = []
        self.v_list = []
        for constraint in constraint_list:
            type = constraint[0]
            dims = constraint[1]
            center = constraint[2]
            radius = constraint[3]

            P = np.zeros((self.transition_dim, self.transition_dim))
            q = np.zeros(self.transition_dim)
            v = radius ** 2
            # P = np.diag(self.a_std ** 2)
            # q = 2 * self.a_std * (self.a_mean - center)
            # v = radius ** 2 - (self.a_mean - center) @ (self.a_mean - center)

            dim_counter = 0
            for dim in dims:
                P[dim, dim] = self.a_std[dim] ** 2
                q[dim] = 2 * self.a_std[dim] * (self.a_mean[dim] - center[dim_counter])
                v -= (self.a_mean[dim] - center[dim_counter]) ** 2
                # if self.scaler is not None:
                #     delta_s = self.scaler.maxs[dim] - self.scaler.mins[dim]
                #     s_min = self.scaler.mins[dim]
                #     P[dim, dim] = delta_s ** 2 / 4
                #     q[dim] = delta_s**2 / 2 + delta_s * (s_min - center[dim_counter])
                #     v -= delta_s**2 / 4 + delta_s * (s_min - center[dim_counter]) + (s_min - center[dim_counter]) ** 2
                # else:
                #     P[dim, dim] = 1
                #     q[dim] = -2 * center[dim_counter]
                #     v -= center[dim_counter] ** 2
                dim_counter += 1

            if type == 'sphere_outside':
                P = -P
                q = -q
                v = -v

            self.P_list.append(P)
            self.q_list.append(q)
            self.v_list.append(v)    

