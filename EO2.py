import numpy as np
import scipy.special

class EO2:
    def __init__(self, func, lb, ub, dim, pop_size, iters,
                 use_temp=True, use_sigmoid=True,
                 use_robin=True, use_levy=True, use_implicit=True):
        self.func = func
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.pop_size = pop_size
        self.iters = iters

        # Ablation experiment switches.
        self.use_temp = use_temp
        self.use_sigmoid = use_sigmoid
        self.use_robin = use_robin
        self.use_levy = use_levy
        self.use_implicit = use_implicit

    def sigmoid(self, x):
        x = np.clip(x, -60, 60)   # Clip x to [-60, 60].
        return 1 / (1 + np.exp(-x))   # Standard sigmoid mapping to [0, 1].



    def levy_flight(self, beta=1.5):
        sigma_u = np.power(
            (scipy.special.gamma(1 + beta) * np.sin(np.pi * beta / 2)) /
            (scipy.special.gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)),
            1 / beta)       # Control the balance between local and global search.
        u = np.random.randn(self.dim) * sigma_u
        v = np.random.randn(self.dim)
        step = u / (np.power(np.abs(v), 1 / beta) + 1e-9)
        return step   # Generate the next jump's step length.

    def compute_sigma(self, gradients, base_sigma=0.1):   # Calculate the perturbation range, or exploration strength.
        return base_sigma * (1 + np.tanh(np.linalg.norm(gradients, axis=1, keepdims=True)))



    # Inputs: the new solution, the difference from the best, and current/maximum iterations.
    def apply_robin_boundary(self, x, grad, iteration, max_iter):
        beta0 = 1.0 - 0.9 * (iteration / max_iter)
        beta1 = 0.1 + 0.9 * (iteration / max_iter)
        norm_grad = np.linalg.norm(grad)
        if norm_grad < 1e-8:
            # Sample a random value from -1 to 1.
            grad_dir = np.random.uniform(-1, 1, size=grad.shape)
            grad_dir /= np.linalg.norm(grad_dir) + 1e-9
        else:
            grad_dir = grad / (norm_grad + 1e-9)
        projected = beta0 * x + beta1 * grad_dir
        # Constrain the generated values to the allowed range.
        return np.clip(projected, self.lb, self.ub)

    def implicit_update(self, history, lambd=0.05):
        T = len(history)
        weights = np.exp(-lambd * np.arange(T))[::-1].reshape(T, 1, 1)
        return np.sum(weights * np.array(history), axis=0) / np.sum(weights)

    def optimize(self, verbose=True):
        X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))      # Initialize a random pop_size-by-dim array within the specified bounds.
        fitness = np.apply_along_axis(self.func, 1, X)   # Evaluate fitness for each row.
        best_idx = np.argmin(fitness)   # Find the index of the minimum value as besti_idx.
        best = X[best_idx].copy()       # Store the best solution.
        best_fit = fitness[best_idx]    # Store the best fitness.

        convergence_curve = []    # Initialize the data.
        position_history = []

        T0 = 1.0             # Initialize the time parameter.
        T_min = 1e-4         # Define the minimum stopping time.

        for t in range(self.iters):          # Exit at the maximum iteration count.
            T = T0 * np.exp(-5 * t / self.iters) + T_min if self.use_temp else 1.0     # Temperature decay model used in optimization.
            GP = np.mean(X, axis=0)       # Calculate the mean of each column of X.
            sorted_idx = np.argsort(fitness)
            pool = [X[sorted_idx[0]], X[sorted_idx[1]], X[sorted_idx[-1]], GP]   # The concentration pool contains the two smallest, the largest, and the mean.
            Ceq = np.mean(pool, axis=0)      # Construct the equilibrium concentration Ceq.

            grads = best - X            # Does this represent the difference from the best solution?
            sigma = self.compute_sigma(grads)    # Calculate exploration strength from best - X.

            for i in range(self.pop_size):      # Iterate over rows.
                a1 = np.random.uniform(1, 2)        # Randomly generate these three variables within their ranges.
                λ = np.random.uniform(0, 1)
                r = np.random.uniform(0, 1)

                F = a1 * np.sign(r - 0.5) * np.exp(-λ * t / self.iters)  # Update coefficient with a modified right-hand side relative to the original formula.
                GCP = 0.5 if np.random.rand() < 0.5 else 1    # GCP is 0.5 or 1 with equal probability; this exploration/exploitation control does not decay with iteration.
                G0 = GCP * (Ceq - λ * X[i])         # Intermediate term based on GCP and the difference between particle and mean concentration positions.
                G = G0 / (np.linalg.norm(G0) + 1e-9) * F      # Final generation rate term.

                diff = best - X[i]    # Calculate the difference from the best solution.
                S = self.sigmoid(10 * diff / (np.linalg.norm(X[i]) + 1e-9)) if self.use_sigmoid else 1.0
                # Apply the sigmoid mapping for normalization.
                # Generate a new particle: the equilibrium term and global exploration term follow EO.
                # The third, exploitation-related term is modified. Does simulated temperature T provide direction?
                X_new = Ceq + (X[i] - Ceq) * F + (G / (λ + 1e-9)) * S * T

                if self.use_levy:
                    # Add Levy flights to improve global search and escape local optima.
                    # Return 0.1 during the first 20% of iterations, otherwise 0.03.
                    levy_scale = 0.1 if t < self.iters * 0.2 else 0.03
                    step = self.levy_flight()
                    step /= np.linalg.norm(step) + 1e-9
                    # Use larger early jumps for exploration and smaller later jumps for exploitation.
                    X_new += step * T * levy_scale

                # Purpose not fully understood; check whether values are valid.
                if self.use_robin:
                    X_new = self.apply_robin_boundary(X_new, grads[i], t, self.iters)
                else:
                    X_new = np.clip(X_new, self.lb, self.ub)
                # Evaluate the new solution's fitness.
                fit_new = self.func(X_new)
                # Greedily replace the previous solution if fitness improves, and update the global best if appropriate.
                if fit_new < fitness[i]:
                    X[i] = X_new
                    fitness[i] = fit_new
                    if fit_new < best_fit:
                        best = X_new.copy()
                        best_fit = fit_new
            # Append X to the history pool; this pool keeps growing, whereas the paper uses only two entries.
            position_history.append(X.copy())
            if self.use_implicit and t > self.iters * 0.3 and len(position_history) > 10:
                X += 0.2 * (self.implicit_update(position_history[-10:]) - X)
            # Is this the historical best?
            convergence_curve.append(np.log10(best_fit + 1e-10))

            if verbose and (t + 1) % 10 == 0:
                print(f"Iter {t+1}/{self.iters} - Best Fitness: {best_fit:.4e}")

        return best, best_fit, convergence_curve

    def run(self, verbose=True):
        return self.optimize(verbose=verbose)
