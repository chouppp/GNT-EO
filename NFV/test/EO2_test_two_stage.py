import numpy as np
import scipy.special

class EO2_twostage:
    def __init__(self, func, lb, ub, dim, pop_size, iters,mapping_table,substrate_nodes,substrate_edges,node_function,conn,virtual_edges,substrate_edges2,
                 use_temp=True, use_sigmoid=True,
                 use_robin=True, use_levy=True, use_implicit=True):
        self.func = func
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.pop_size = pop_size
        self.iters = iters
        self.mapping_table=mapping_table
        self.substrate_nodes=substrate_nodes
        self.substrate_edges=substrate_edges
        self.node_function=node_function
        self.conn=conn
        self.virtual_edges=virtual_edges
        self.substrate_edges2=substrate_edges2

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

    # Convert the node mapping into a link-mapping plan.
    def convert_virtual_links_to_substrate_links(self, decode):
        """vlink_list: [{'source': v1, 'target': v2, 'bw': x, 'max_bw': y}, ...].
                decode_table: {virtual_node: substrate_node}.
                Return a link-mapping table containing only source, target, and bw."""
        slink_list = []
        for link in self.virtual_edges:
            v_src = link["source"]
            v_dst = link["target"]
            bw = link["bw"]

            # Raise an error and stop if a node is missing from the mapping.
            if v_src not in decode or v_dst not in decode:
                print(f"❌ 映射失败：虚拟节点 {v_src} 或 {v_dst} 不存在于 decode_table 中！")

            s_src = decode[v_src]
            s_dst = decode[v_dst]

            # Add to the new physical-link mapping table.
            slink_list.append({
                "source": s_src,
                "target": s_dst,
                "bw": bw
            })

        return slink_list

    # Map the probability distribution back to node assignments.
    def trans_fur(self,best_X, mapping_table):
        """best_X: optimizer solution, e.g. [0.17, 0.44, 0.56, 0.57, 0.44].
                mapping_table: virtual-node ID -> intervals.
                Return a dictionary such as {0: 8, 1: 16, 2: 20, ...}."""
        decoded = {}

        for v_id, r in enumerate(best_X):

            # Handle r == 1.0.
            if r > 1.0:
                r = 1.0

            intervals = mapping_table[v_id]["intervals"]

            mapped_node = None
            for s_node, (low, high) in intervals.items():
                if low <= r <= high:
                    mapped_node = s_node
                    break

            decoded[v_id] = mapped_node

        return decoded

    # Check for duplicate randomly selected nodes.
    def has_duplicate_mappings(self,decoded_dict):
        """
        decoded_dict: {v_id: s_node}
        """
        used_nodes = list(decoded_dict.values())
        return len(used_nodes) != len(set(used_nodes))

    # Ensure the generated initial solution is feasible.
    def repair_solution(self, X_row, mapping_table, lb, ub):
        """X_row: probability-valued solution, e.g. [0.12, 0.55, 0.88].
                mapping_table: interval mapping table; lb/ub are bounds, usually 0/1.
                Return (X_row, decoded) for a conflict-free mapping, or (False, None) after repeated failures."""
        MAX_TRY = 70

        # 1. Try the original solution first.
        decoded = self.trans_fur(X_row, mapping_table)
        if not self.has_duplicate_mappings(decoded):
            return X_row, decoded

        # 2. Attempt random repair up to MAX_TRY times.
        for _ in range(MAX_TRY):
            new_row = np.random.uniform(lb, ub, len(X_row))
            decoded = self.trans_fur(new_row, mapping_table)

            if not self.has_duplicate_mappings(decoded):
                return new_row, decoded

        # 3. If all attempts fail, classify the solution as unmappable.
        print("❌ repair_solution：在最大尝试次数内未找到无冲突映射，判定该虚拟链路不可映射")
        return False, None

    def generate_init_list(self):
        X = np.zeros((self.pop_size, self.dim))
        decoded_list = []

        for i in range(self.pop_size):
            row = np.random.uniform(self.lb, self.ub, self.dim)

            row_fixed, decoded = self.repair_solution(
                row, self.mapping_table, self.lb, self.ub
            )

            # Core fix: return False immediately when repair fails.
            if row_fixed is False:
                print("❌ generate_init_list：存在无法修复的初始解，判定该 VNR 不可映射")
                return False, None

            X[i] = row_fixed
            decoded_list.append(decoded)

        return X, decoded_list


    # Inspect resource consumption of the generated link mapping.
    def build_changed_edges(self,mapped_links):
        """Return changed_edges: {(u,v): consumed_bw}."""
        changed_edges = {}

        for item in mapped_links:
            bw = item["bw"]
            link_edges = item["linkpath"]  # [[u,v], [v,w], ...]

            for (u, v) in link_edges:
                key = (u, v)

                if key not in changed_edges:
                    changed_edges[key] = 0.0

                changed_edges[key] += bw

        return changed_edges

    # Validate feasibility.
    def check_mapped_links_feasible(self,mapped_links):
        changed_edges = self.build_changed_edges(mapped_links)

        for edge, used_bw in changed_edges.items():
            remain_bw = self.substrate_edges.get(edge, None)

            if remain_bw is None:
                print(f"❌ 非法解：单跳链路 {edge} 不存在于 s_edges 中")
                return False, changed_edges

            if used_bw > remain_bw:
                print(
                    f"❌ 非法解：单跳链路 {edge} 超载，"
                    f"累计需求={used_bw}, 当前剩余={remain_bw}"
                )
                return False, changed_edges

        return True, changed_edges

    # Check link-mapping feasibility from the decoded solution.
    def map_slinks_to_paths_numpy(self, slink_list):
        results = []
        # Sort by bandwidth first.
        slink_list = sorted(slink_list, key=lambda x: x["bw"], reverse=True)
        # Temporary copy of the current global residual bandwidth.
        temp_edge_bw = self.substrate_edges.copy()

        for sl in slink_list:
            src = sl["source"]
            dst = sl["target"]
            bw_required = sl["bw"]
            key = (src, dst)
            # 1. Handle missing paths for (src,dst).
            if key not in self.substrate_edges2:
                print(f"❌ 未找到链路 {src} → {dst}")
                return False  # Fail immediately.
            hop_arr = self.substrate_edges2[key]["hop"]  # NumPy array
            bw_arr = self.substrate_edges2[key]["bw"]  # NumPy array
            path_list = self.substrate_edges2[key]["path"]
            # 2. Find bandwidth-feasible paths.
            valid_idx = np.where(bw_arr >= bw_required)[0]
            if valid_idx.size == 0:
                print(f"❌ 带宽不足 {src} → {dst} (需要 {bw_required})")
                return False  # Fail immediately.
            # 3. Since hops are sorted, choose the feasible path with the fewest hops.
            i = valid_idx[0]

            result_item = {
                "source": src,
                "target": dst,
                "bw": bw_required,
                "hop": hop_arr[i],
                "linkpath": path_list[i]
            }

            results.append(result_item)

        # 4. All virtual links have been mapped successfully.
        return results

    def map_slinks_to_paths_numpy_test(self, slink_list):
        results = []

        # Temporary copy of the current global residual bandwidth.
        temp_edge_bw = self.substrate_edges.copy()

        for sl in slink_list:
            src = sl["source"]
            dst = sl["target"]
            bw_required = sl["bw"]
            key = (src, dst)

            if key not in self.substrate_edges2:
                print(f"❌ 未找到链路 {src} → {dst}")
                return False

            hop_arr = self.substrate_edges2[key]["hop"]
            path_list = self.substrate_edges2[key]["path"]

            chosen_idx = None
            chosen_bottleneck = None

            # Search for feasible paths in ascending hop order.
            for i, path in enumerate(path_list):
                path_bottleneck = float("inf")
                feasible = True

                for u, v in path:
                    edge_key = (u, v)
                    remain_bw = temp_edge_bw.get(edge_key, 0)

                    if remain_bw < bw_required:
                        feasible = False
                        break

                    path_bottleneck = min(path_bottleneck, remain_bw)

                if feasible:
                    chosen_idx = i
                    chosen_bottleneck = path_bottleneck
                    break

            if chosen_idx is None:
                print(f"❌ 带宽不足 {src} → {dst} (需要 {bw_required})")
                return False

            chosen_path = path_list[chosen_idx]

            # Deduct temporary bandwidth.
            for u, v in chosen_path:
                temp_edge_bw[(u, v)] -= bw_required

            results.append({
                "source": src,
                "target": dst,
                "bw": bw_required,
                "hop": hop_arr[chosen_idx],
                "path_bottleneck": chosen_bottleneck,
                "linkpath": chosen_path
            })

        return results

    def optimize(self, verbose=True):
        # X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))  # Initialize a bounded pop_size-by-dim array.
        X, decoded_list = self.generate_init_list()
        # Return False and reject the VNR if initial-solution generation fails.
        if X is False:
            print("❌ 初始解生成失败，判定该虚拟网络不可映射")
            return False, None, None, None  # Or use the return format expected by run().
        fitness = np.apply_along_axis(self.func, 1, X,self.mapping_table,self.substrate_nodes,self.substrate_edges,self.node_function)   # Evaluate fitness for each row.
        best_idx = np.argmax(fitness)   # Find the index of the minimum value as besti_idx.
        best = X[best_idx].copy()       # Store the best solution.
        best_fit = fitness[best_idx]    # Store the best fitness.

        convergence_curve = []    # Initialize the data.
        position_history = []
        prev_best_list={}       # Record the best node mapping.
        prev_best_slink={}
        prev_best_slink_result = {}


        T0 = 1.0                                                                            # Initialize the time parameter.
        T_min = 1e-4                                                                        # Define the minimum stopping time.
        found_feasible = False
        best_link_result = None

        tabu_best_list = set()  # Blacklist containing tuple(best_list).

        for t in range(self.iters):                                                        # Exit at the maximum iteration count.
            T = T0 * np.exp(-5 * t / self.iters) + T_min if self.use_temp else 1.0         # Temperature decay model used in optimization.
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

                X_check=self.trans_fur(X_new,self.mapping_table)
                #s_link=self.convert_virtual_links_to_substrate_links(X_check)
                #s_link_result=self.map_slinks_to_paths_batch(s_link)
                if  self.has_duplicate_mappings(X_check) :
                    # Discard invalid solutions and skip this iteration.
                    continue

                # 2. Discard blacklisted solutions immediately.
                if tuple(X_check) in tabu_best_list:
                    continue
                    # Evaluate the new solution's fitness.
                fit_new = self.func(X_new,self.mapping_table,self.substrate_nodes,self.substrate_edges,self.node_function)

                # Greedily replace the previous solution if fitness improves, and update the global best if appropriate.
                if fit_new > fitness[i]:
                    X[i] = X_new
                    fitness[i] = fit_new
                    if fit_new > best_fit:
                        best = X_new.copy()
                        best_fit = fit_new
            # Append X to the history pool; this pool keeps growing, whereas the paper uses only two entries.
            position_history.append(X.copy())
            if self.use_implicit and t > self.iters * 0.3 and len(position_history) > 10:
                X += 0.2 * (self.implicit_update(position_history[-10:]) - X)
            # Is this the historical best?
            convergence_curve.append(np.log10(best_fit + 1e-10))

            if verbose and (t + 1) % 10 == 0:
                best_list=self.trans_fur(best,self.mapping_table)
                best_key = tuple(best_list)
                # Step 0: a blacklisted current best has already failed validation.
                if best_key in tabu_best_list:
                    print(f"Iter {t + 1}/{self.iters} 当前 best 命中黑名单，跳过该解")
                    continue

                # Step 1: check whether it matches the previous solution.
                if (
                        prev_best_list is not None
                        and best_list == prev_best_list
                        and tuple(prev_best_list) not in tabu_best_list
                        and prev_best_slink_result is not None
                ):
                    s_link = prev_best_slink
                    s_link_result = prev_best_slink_result
                    print(f"Iter {t+1}/{self.iters} - Best Fitness: {best_fit:}-best：{best_list}\n best-link:{s_link_result}")
                else:
                    # Step 2: remap a new best_list.
                    s_link = self.convert_virtual_links_to_substrate_links(best_list)
                    s_link_result = self.map_slinks_to_paths_numpy_test(s_link)
                    # Discard the solution if link mapping fails.
                    if s_link_result is False:
                        print(f"Iter {t + 1}/{self.iters} 本轮映射失败，加入黑名单: {best_list}")
                        tabu_best_list.add(best_key)

                        # Reduce the current best's fitness so it is not selected again next iteration.
                        for j in range(self.pop_size):
                            if tuple(self.trans_fur(X[j], self.mapping_table)) == best_key:
                                fitness[j] = -1e12

                        # Select a new best from the current population.
                        new_best_idx = np.argmax(fitness)
                        best = X[new_best_idx].copy()
                        best_fit = fitness[new_best_idx]

                        continue
                    feasible, changed_edges = self.check_mapped_links_feasible(s_link_result)
                    if not feasible:
                        print("❌ 当前解不合法，舍弃该映射结果")
                        tabu_best_list.add(best_key)

                        for j in range(self.pop_size):
                            if tuple(self.trans_fur(X[j], self.mapping_table)) == best_key:
                                fitness[j] = -1e12

                        new_best_idx = np.argmax(fitness)
                        best = X[new_best_idx].copy()
                        best_fit = fitness[new_best_idx]

                        continue

                    found_feasible = True
                    # Update the cache.
                    prev_best_list = best_list
                    prev_best_slink = s_link
                    prev_best_slink_result = s_link_result
                    print(f"Iter {t + 1}/{self.iters} - Best Fitness: {best_fit:}-best：{best_list}\n best-link:{s_link_result}")

        # End of the loop.
        if not found_feasible:
            print("❌ 未找到任何可行映射")
            return False, None, None, None
        return prev_best_list, best_fit, convergence_curve,prev_best_slink_result

    def run(self, verbose=True):
        return self.optimize(verbose=verbose)
