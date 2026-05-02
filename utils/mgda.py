import torch
import numpy as np
from scipy.optimize import minimize

class MGDAOptimizer:
    """
    Multiple Gradient Descent Algorithm (MGDA) using SLSQP to find the Pareto optimal weights.
    Finds weights w_i that minimize || \sum_i w_i g_i ||^2 subject to w_i >= 0, \sum w_i = 1.
    """
    @staticmethod
    def get_optimal_weights(grads):
        """
        grads: list of 1D PyTorch tensors representing the flattened gradients for each loss.
        """
        # Stack gradients into a matrix (num_tasks, num_parameters)
        M = torch.stack(grads)
        
        # Compute the Gram matrix (num_tasks, num_tasks)
        # Gram_ij = dot(g_i, g_j)
        Gram = torch.matmul(M, M.T).cpu().numpy()
        
        # Objective function: w^T * Gram * w
        def objective(w):
            return np.dot(w.T, np.dot(Gram, w))
            
        # Gradient of objective: 2 * Gram * w
        def jacobian(w):
            return 2 * np.dot(Gram, w)
            
        num_tasks = len(grads)
        
        # Initial guess: equal weights
        w0 = np.ones(num_tasks) / num_tasks
        
        # Bounds: 0 <= w_i <= 1
        bounds = [(0.0, 1.0) for _ in range(num_tasks)]
        
        # Constraint: sum(w) = 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        
        # Solve using SLSQP
        res = minimize(objective, w0, method='SLSQP', jac=jacobian, bounds=bounds, constraints=constraints)
        
        # Return the optimal weights as a tensor
        return torch.FloatTensor(res.x).to(grads[0].device)
