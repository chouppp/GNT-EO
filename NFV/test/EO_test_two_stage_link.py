import json
import time
from collections import deque

import numpy as np
import scipy.special


class EO2_twostage:
    # Share a lazy path cache for the same large substrate_edges2 object.
    # Keep a source reference in each entry to prevent false hits from Python object-ID reuse.
    _shared_path_cache_entries = {}

    def __init__(self, func, lb, ub, dim, pop_size, iters, mapping_table, substrate_nodes, substrate_edges,
                 node_function, conn, virtual_edges, substrate_edges2, domain_temperatures, domains,
                 use_temp=True, use_sigmoid=True,
                 use_robin=True, use_levy=True, use_implicit=True,
                 record_iteration_history=True,
                 iteration_history_path="iteration_history.jsonl", vnr_id=None,
                 use_fitness_cache=True,
                 fitness_uses_decoded_mapping=False):
        self.fitness_uses_decoded_mapping = fitness_uses_decoded_mapping
        self.func = func
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.pop_size = pop_size
        self.iters = iters
        self.mapping_table = mapping_table
        self.substrate_nodes = substrate_nodes
        self.substrate_edges = substrate_edges
        self.node_function = node_function
        self.conn = conn
        self.virtual_edges = virtual_edges
        self.substrate_edges2 = substrate_edges2

        # Ablation experiment switches.
        self.use_temp = use_temp
        self.use_sigmoid = use_sigmoid
        self.use_robin = use_robin
        self.use_levy = use_levy
        self.use_implicit = use_implicit
        self.domain_temperatures = domain_temperatures
        self.domains = domains
        # Write one JSONL record per completed VNR containing all iteration details.
        self.record_iteration_history = record_iteration_history
        self.iteration_history_path = iteration_history_path
        self.vnr_id = vnr_id
        self.use_fitness_cache = use_fitness_cache

        # Cache values that stay fixed during a run to avoid repeated calculations and database queries.
        self._node_rank_cache = None
        self._virtual_degrees_cache = self.build_virtual_degrees(self.virtual_edges)
        self._interval_cache = self._prepare_interval_cache()
        # Cache paths on demand without copying or traversing all of substrate_edges2.
        # Reuse processed paths across optimizer instances sharing substrate_edges2.
        self._path_cache = self._get_shared_path_cache()
        self._levy_sigma_cache = {}
        self._fitness_cache = {}
        self.verbose = False

    def _prepare_interval_cache(self):
        """Preprocess interval iteration data while preserving the original intervals order."""
        mapping_items = (
            self.mapping_table.items()
            if isinstance(self.mapping_table, dict)
            else enumerate(self.mapping_table)
        )
        return {
            v_id: [
                (s_node, low, high)
                for s_node, (low, high) in info["intervals"].items()
            ]
            for v_id, info in enumerate(self.mapping_table)
            for v_id, info in mapping_items
        }

    def _get_shared_path_cache(self):
        """Get the shared path cache for the current substrate_edges2 object without copying source data."""
        cache_key = id(self.substrate_edges2)
        entry = self._shared_path_cache_entries.get(cache_key)

        if entry is None or entry["source"] is not self.substrate_edges2:
            entry = {
                "source": self.substrate_edges2,
                "cache": {}
            }
            self._shared_path_cache_entries[cache_key] = entry

        return entry["cache"]

    @classmethod
    def clear_shared_path_cache(cls, substrate_edges2=None):
        """Call explicitly to release the shared path cache; unnecessary during normal execution."""
        if substrate_edges2 is None:
            cls._shared_path_cache_entries.clear()
        else:
            cls._shared_path_cache_entries.pop(id(substrate_edges2), None)

    def _get_path_info(self, key):
        """Lazily preprocess path data for one source/destination pair."""
        path_info = self._path_cache.get(key)
        if path_info is not None:
            return path_info

        raw_info = self.substrate_edges2.get(key)
        if raw_info is None:
            return None

        path_info = {
            "hop": raw_info["hop"],
            "path": raw_info["path"],
            "edge_path": [
                [(u, v) for u, v in path]
                for path in raw_info["path"]
            ]
        }
        self._path_cache[key] = path_info
        return path_info

    def _mapping_key(self, decoded):
        """Generate a stable cache key for identical node mappings."""
        return tuple(decoded[v_id] for v_id in range(self.dim))

    def _evaluate_fitness(self, decoded, s_link, s_link_result, virtual_degrees):
        key = self._mapping_key(decoded)
        if self.use_fitness_cache and key in self._fitness_cache:
            return self._fitness_cache[key]
        value = self.func(
            s_link=decoded if self.fitness_uses_decoded_mapping else s_link,
            substrate_nodes=self.substrate_nodes, substrate_edges=self.substrate_edges,
            virtual_degrees=virtual_degrees, s_link_result=s_link_result,
            domain_temperatures=self.domain_temperatures, domains=self.domains)
        if self.use_fitness_cache:
            self._fitness_cache[key] = value
        return value

    def _normalize_temperature(self, value):
        """Used only for EO2 search coefficients; pass raw domain temperatures directly to fitness evaluation."""
        values = [abs(float(d['T'])) for d in (self.domain_temperatures or [])
                  if d.get('T') is not None and np.isfinite(float(d['T']))]
        self._temperature_scale = max(1.0, float(np.median(values))) if values else 1.0
        return float(0.5 * (1.0 + np.tanh(value / self._temperature_scale)))

    def compute_average_network_temperature(self):
        """Clip each domain to [-10000, 10000] before averaging; null and missing domains count as zero."""
        temperatures = {
            int(item["domain_id"]): item.get("T")
            for item in (self.domain_temperatures or [])
        }
        if self.domains is None:
            domain_ids = list(temperatures)
        elif isinstance(self.domains, dict):
            domain_ids = list(self.domains)
        else:
            domain_ids = range(len(self.domains))
        values = []
        for domain_id in domain_ids:
            value = temperatures.get(domain_id)
            if value is None:
                values.append(0.0)
                continue
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"域 {domain_id} 的温度必须是有限数值。")
            values.append(float(np.clip(value, -10000.0, 10000.0)))
        return float(np.mean(values)) if values else 0.0

    def write_iteration_history(self, record):
        """Append one complete EO2 run record in JSONL format."""
        if not self.record_iteration_history or not self.iteration_history_path:
            return
        with open(self.iteration_history_path, "a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    def sigmoid(self, x):
        x = np.clip(x, -60, 60)  # Clip x to [-60, 60].
        return 1 / (1 + np.exp(-x))  # Standard sigmoid mapping to [0, 1].

    def levy_flight(self, beta=1.5):
        sigma_u = self._levy_sigma_cache.get(beta)
        if sigma_u is None:
            sigma_u = np.power(
                (scipy.special.gamma(1 + beta) * np.sin(np.pi * beta / 2)) /
                (scipy.special.gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)),
                1 / beta)
            self._levy_sigma_cache[beta] = sigma_u
        u = np.random.randn(self.dim) * sigma_u
        v = np.random.randn(self.dim)
        step = u / (np.power(np.abs(v), 1 / beta) + 1e-9)
        return step  # Generate the next jump's step length.

    def compute_sigma(self, gradients, base_sigma=0.1):  # Calculate the perturbation range, or exploration strength.
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
                # Preserve subsequent access and exception behavior; do not emit intermediate logs during iterations.
                pass

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
    def trans_fur(self, best_X, mapping_table):
        """best_X: optimizer solution, e.g. [0.17, 0.44, 0.56, 0.57, 0.44].
                mapping_table: virtual-node ID -> intervals.
                Return a dictionary such as {0: 8, 1: 16, 2: 20, ...}."""
        decoded = {}

        for v_id, r in enumerate(best_X):

            # Handle r == 1.0.
            if r > 1.0:
                r = 1.0

            mapped_node = None
            for s_node, low, high in self._interval_cache[v_id]:
                if low <= r <= high:
                    mapped_node = s_node
                    break

            decoded[v_id] = mapped_node

        return decoded

    # Check for duplicate randomly selected nodes.
    def has_duplicate_mappings(self, decoded_dict):
        """
        decoded_dict: {v_id: s_node}
        """
        used_nodes = list(decoded_dict.values())
        return len(used_nodes) != len(set(used_nodes))

    # Ensure the generated initial solution is feasible.
    def repair_solution(self, X_row, mapping_table, lb, ub, max_try=50):
        """Check the current solution first.
                Retry duplicate-mapping repair a bounded number of times to avoid an infinite loop."""
        best_row = X_row
        best_decoded = self.trans_fur(X_row, mapping_table)

        for _ in range(max_try):
            decoded = self.trans_fur(X_row, mapping_table)

            # 1. Nodes must be distinct.
            if not self.has_duplicate_mappings(decoded):
                slink = self.convert_virtual_links_to_substrate_links(decoded)
                link_map = self.map_slinks_to_paths_numpy_test(slink)

                # Return directly here if link_map validation is temporarily disabled.
                return X_row, decoded

            # 2. Regenerate when conflicts occur.
            if self.use_temp:
                X_row = self.generate_temp_guided_row()
            else:
                X_row = np.random.uniform(lb, ub, len(X_row))

            best_row = X_row
            best_decoded = self.trans_fur(X_row, mapping_table)

        # Return the last attempt after the repair limit to avoid an infinite loop.
        return best_row, best_decoded

    # Get the node resource matrix.
    def build_node_adjacent_bw(self, num_nodes=25):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT source_node, destination_node, bw_resource
            FROM link_single_table
        """)
        rows = cursor.fetchall()
        node_adj_bw = np.zeros(num_nodes, dtype=np.float32)
        if not rows:
            return node_adj_bw
        rows_np = np.array(rows, dtype=object)
        src_arr = rows_np[:, 0].astype(np.int32)
        dst_arr = rows_np[:, 1].astype(np.int32)
        bw_arr = rows_np[:, 2].astype(np.float32)

        # Accumulate at the source node.
        np.add.at(node_adj_bw, src_arr, bw_arr)
        # Accumulate at the destination node.
        np.add.at(node_adj_bw, dst_arr, bw_arr)

        return node_adj_bw

    def build_domain_node_rank_test(self):
        node_rank = {}
        # Assume node IDs run from 0 to num_nodes - 1.
        num_nodes = max(self.substrate_nodes) + 1
        node_adj_bw = self.build_node_adjacent_bw()
        # Without temperature guidance, sort nodes by descending residual incident-link bandwidth.
        if (not self.use_temp) or (self.domain_temperatures is None) or (self.domains is None):
            sorted_nodes = sorted(
                self.substrate_nodes,
                key=lambda n: node_adj_bw[n],
                reverse=True
            )

            for rank, node in enumerate(sorted_nodes):
                node_rank[node] = rank

            return node_rank

        domain_list = []

        for item in self.domain_temperatures:
            d_id = item["domain_id"]
            T = item["T"]

            # Place None temperatures last.
            T_val = float("inf") if T is None else T

            # Get nodes in this domain.
            nodes = self.domains[d_id]

            # Within a domain, sort by descending residual incident-link bandwidth.
            sorted_nodes = sorted(
                nodes,
                key=lambda n: node_adj_bw[n],
                reverse=True
            )

            domain_list.append((T_val, sorted_nodes))

        # Sort domains by ascending temperature, prioritizing cooler domains.
        domain_list.sort(key=lambda x: x[0])

        rank = 0
        for _, nodes in domain_list:
            for node in nodes:
                if node not in node_rank:
                    node_rank[node] = rank
                    rank += 1

        # Handle nodes missing from all domains.
        remaining_nodes = [node for node in self.substrate_nodes if node not in node_rank]

        # Append omitted nodes in descending residual incident-link bandwidth order.
        remaining_nodes.sort(key=lambda n: node_adj_bw[n], reverse=True)

        for node in remaining_nodes:
            node_rank[node] = rank
            rank += 1

        return node_rank

    def generate_temp_guided_row(self):

        X_row = np.zeros(self.dim)
        used_nodes = set()

        # Node bandwidth ranking is constant within one optimization run; compute it once.
        # Avoid repeating SQL queries for every candidate solution.
        if self._node_rank_cache is None:
            self._node_rank_cache = self.build_domain_node_rank_test()
        node_rank = self._node_rank_cache

        for j in range(self.dim):
            info = self.mapping_table[j]
            candidates = info["candidates"]
            intervals = info["intervals"]

            ranked_candidates = []
            for node in candidates:
                rank = node_rank.get(node, 10 ** 9)
                ranked_candidates.append((rank, node))

            ranked_candidates.sort(key=lambda x: x[0])

            # Change: choose randomly among the top k candidates.
            k = min(5, len(ranked_candidates))  # Tunable range: 3 to 5.
            candidate_pool = [node for _, node in ranked_candidates[:k]]

            # Exclude already used nodes first.
            candidate_pool = [n for n in candidate_pool if n not in used_nodes]

            if len(candidate_pool) == 0:
                candidate_pool = [node for _, node in ranked_candidates[:k]]

            chosen_node = np.random.choice(candidate_pool)

            used_nodes.add(chosen_node)

            left, right = intervals[chosen_node]

            eps = 1e-8
            l = max(self.lb[j], left + eps)
            r = min(self.ub[j], right - eps)

            if l >= r:
                X_row[j] = (left + right) / 2.0
            else:
                X_row[j] = np.random.uniform(l, r)

        return X_row

    def generate_init_list(self):
        X = []
        decoded_list = []
        slink_list = []
        slink_result_list = []

        max_try = self.pop_size * 10
        try_cnt = 0

        target_temp_num = 3  # Three temperature-guided solutions.
        target_rand_num = 2  # Two random solutions.

        def try_add_solution(row):
            if row is False:
                return False

            decoded = self.trans_fur(row, self.mapping_table)

            # Check for duplicate nodes.
            if self.has_duplicate_mappings(decoded):
                return False

            # Link mapping.
            s_link = self.convert_virtual_links_to_substrate_links(decoded)
            s_link_result = self.map_slinks_to_paths_numpy_test(s_link)

            if s_link_result is False:
                return False

            feasible, _ = self.check_mapped_links_feasible(s_link_result)
            if not feasible:
                return False

            X.append(row.copy())
            decoded_list.append(decoded.copy() if isinstance(decoded, list) else decoded)
            slink_list.append(s_link.copy() if isinstance(s_link, dict) else s_link)
            slink_result_list.append(
                [item.copy() for item in s_link_result] if isinstance(s_link_result, list) else s_link_result
            )
            return True

        # Stage 1: generate three temperature-guided solutions.
        temp_cnt = 0
        while temp_cnt < target_temp_num and try_cnt < max_try:
            row = self.generate_temp_guided_row()
            try_cnt += 1
            if try_add_solution(row):
                temp_cnt += 1

        # Stage 2: generate two random solutions.
        rand_cnt = 0
        while rand_cnt < target_rand_num and try_cnt < max_try * 2:
            row = np.random.uniform(self.lb, self.ub, self.dim)
            try_cnt += 1
            if try_add_solution(row):
                rand_cnt += 1

        # Complete failure.
        if len(X) == 0:
            return False, None, None, None

        # If too few solutions exist, copy the current best to fill the population.
        if len(X) < self.pop_size:
            virtual_degrees = self._virtual_degrees_cache

            fitness_list = []
            for i in range(len(X)):
                fit_val = self._evaluate_fitness(
                    decoded_list[i],
                    slink_list[i],
                    slink_result_list[i],
                    virtual_degrees
                )
                fitness_list.append(fit_val)

            # The current objective is a cost: smaller is better.
            best_idx = int(np.argmin(fitness_list))

            best_row = X[best_idx].copy()
            best_decoded = decoded_list[best_idx].copy() if isinstance(decoded_list[best_idx], list) else decoded_list[
                best_idx]
            best_slink = slink_list[best_idx].copy() if isinstance(slink_list[best_idx], dict) else slink_list[best_idx]
            best_slink_result = [item.copy() for item in slink_result_list[best_idx]]

            while len(X) < self.pop_size:
                X.append(best_row.copy())
                decoded_list.append(best_decoded.copy() if isinstance(best_decoded, list) else best_decoded)
                slink_list.append(best_slink.copy() if isinstance(best_slink, dict) else best_slink)
                slink_result_list.append([item.copy() for item in best_slink_result])

        return np.array(X), decoded_list, slink_list, slink_result_list

    # Inspect resource consumption of the generated link mapping.
    def build_changed_edges(self, mapped_links):
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

    # Get virtual-network degrees.
    def build_virtual_degrees(self, v_edges):
        degrees = {}

        for edge in v_edges:
            u = edge["source"]
            v = edge["target"]

            degrees[u] = degrees.get(u, 0) + 1
            degrees[v] = degrees.get(v, 0) + 1

        return degrees

    # Validate feasibility.
    def check_mapped_links_feasible(self, mapped_links):
        changed_edges = self.build_changed_edges(mapped_links)

        for edge, used_bw in changed_edges.items():
            remain_bw = self.substrate_edges.get(edge, None)

            if remain_bw is None:
                return False, changed_edges

            if used_bw > remain_bw:
                return False, changed_edges

        return True, changed_edges

    # Check link-mapping feasibility from the decoded solution.
    def map_slinks_to_paths_numpy_test(self, slink_list):
        results = []

        # Track only edges consumed by this mapping, equivalent to copying all bandwidths and deducting per edge.
        used_edge_bw = {}

        for sl in slink_list:
            src = sl["source"]
            dst = sl["target"]
            bw_required = sl["bw"]
            key = (src, dst)

            path_info = self._get_path_info(key)
            if path_info is None:
                return False

            hop_arr = path_info["hop"]
            path_list = path_info["path"]
            edge_path_list = path_info["edge_path"]

            chosen_idx = None
            chosen_bottleneck = None

            # Search for feasible paths in ascending hop order.
            for i, edge_path in enumerate(edge_path_list):
                path_bottleneck = float("inf")
                feasible = True

                for u, v in edge_path:
                    edge_key = (u, v)
                    remain_bw = (
                        self.substrate_edges.get(edge_key, 0)
                        - used_edge_bw.get(edge_key, 0.0)
                    )

                    if remain_bw < bw_required:
                        feasible = False
                        break

                    path_bottleneck = min(path_bottleneck, remain_bw)

                if feasible:
                    chosen_idx = i
                    chosen_bottleneck = path_bottleneck
                    break

            if chosen_idx is None:
                return False

            chosen_path = path_list[chosen_idx]

            # Deduct temporary bandwidth.
            for u, v in chosen_path:
                edge_key = (u, v)
                used_edge_bw[edge_key] = used_edge_bw.get(edge_key, 0.0) + bw_required

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
        run_start_time = time.perf_counter()
        self.verbose = verbose
        # Use the supplied temperature snapshot; normalize only the initial search temperature T0.
        self.average_network_temperature = self.compute_average_network_temperature()
        T0 = self._normalize_temperature(self.average_network_temperature)
        T_min = 1e-4
        T = T0 + T_min if self.use_temp else 1.0
        if verbose:
            print(
                f"EO2 temperature: average_network_temperature={self.average_network_temperature}, "
                f"update_temperature={T}"
            )

        # Refresh input-dependent caches for each run to preserve behavior across runs.
        self._node_rank_cache = None
        self._virtual_degrees_cache = self.build_virtual_degrees(self.virtual_edges)
        self._interval_cache = self._prepare_interval_cache()
        # Do not reset _path_cache; reuse paths already processed lazily since initialization.
        self._fitness_cache = {}

        iteration_records = []
        slink_list = [None] * self.pop_size
        slink_result_list = [None] * self.pop_size
        # X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))  # Initialize a bounded pop_size-by-dim array.
        X, decoded_list, slink_list, slink_result_list = self.generate_init_list()
        # Return False and reject the VNR if initial-solution generation fails.
        if X is False:
            self.last_total_iterations = 0
            self.last_improvement_iteration = None
            self.global_best_update_count = 0
            self.time_to_final_best_seconds = None
            self.last_runtime_seconds = time.perf_counter() - run_start_time
            self.write_iteration_history({
                "optimizer": "EO2",
                "temperature_scale": self._temperature_scale,
                "average_network_temperature": self.average_network_temperature,
                "update_temperature": float(T),
                "vnr_id": self.vnr_id,
                "success": False,
                "total_iterations": 0,
                "final_fitness": None,
                "last_improvement_iteration": None,
                "global_best_update_count": 0,
                "time_to_final_best_seconds": None,
                "total_runtime_seconds": self.last_runtime_seconds,
                "iterations": []
            })
            if verbose:
                print("EO2 finished: no feasible initial solution.")
            return False, None, None, None  # Or use the return format expected by run().
        # Initialize fitness.
        fitness = np.full(self.pop_size, 1e12, dtype=float)
        # For each initial individual: decode nodes, map links, then evaluate jointly.
        virtual_degrees = self._virtual_degrees_cache
        for i in range(len(X)):
            s_link = slink_list[i]
            s_link_result = slink_result_list[i]

            fit_val = self._evaluate_fitness(
                decoded_list[i],
                s_link,
                s_link_result,
                virtual_degrees
            )
            fitness[i] = fit_val

        # fitness = np.apply_along_axis(self.func, 1, X,self.mapping_table,self.substrate_nodes,self.substrate_edges,self.node_function)  # Evaluate each row's fitness.
        best_idx = np.argmin(fitness)  # Find the index of the minimum value as besti_idx.
        best = X[best_idx].copy()  # Store the best solution.
        best_list = self.trans_fur(best, self.mapping_table)
        best_fit = fitness[best_idx]  # Store the best fitness.
        # Record the initial population's best as iteration 0; update only when the global best improves.
        last_improvement_iteration = 0
        global_best_update_count = 0
        time_to_final_best_seconds = 0.0
        self.last_improvement_iteration = 0
        self.global_best_update_count = 0
        self.time_to_final_best_seconds = 0.0
        convergence_curve = []  # Initialize the data.
        position_history = deque(maxlen=10)
        prev_best_list = best_list  # Record the best node mapping.
        prev_best_slink = slink_list[best_idx]
        prev_best_slink_result = slink_result_list[best_idx]
        # generate_init_list already ensures initial population feasibility.
        found_feasible = True
        best_link_result = None

        tabu_best_list = set()  # Blacklist containing tuple(best_list).

        for t in range(self.iters):  # Exit at the maximum iteration count.
            iteration_start_time = time.perf_counter()
            improved_this_iteration = False
            T = T0 * np.exp(-5 * t / self.iters) + T_min if self.use_temp else 1.0
            GP = np.mean(X, axis=0)  # Calculate the mean of each column of X.
            sorted_idx = np.argsort(fitness)
            pool = [X[sorted_idx[0]], X[sorted_idx[1]], X[sorted_idx[-1]], GP]  # The concentration pool contains the two smallest, the largest, and the mean.
            Ceq = np.mean(pool, axis=0)  # Construct the equilibrium concentration Ceq.

            grads = best - X  # Does this represent the difference from the best solution?
            for i in range(self.pop_size):  # Iterate over rows.
                a1 = np.random.uniform(1, 2)  # Randomly generate these three variables within their ranges.
                λ = np.random.uniform(0, 1)
                r = np.random.uniform(0, 1)

                F = a1 * np.sign(r - 0.5) * np.exp(-λ * t / self.iters)  # Update coefficient with a modified right-hand side relative to the original formula.
                GCP = 0.5 if np.random.rand() < 0.5 else 1  # GCP is 0.5 or 1 with equal probability; this exploration/exploitation control does not decay with iteration.
                G0 = GCP * (Ceq - λ * X[i])  # Intermediate term based on GCP and the difference between particle and mean concentration positions.
                G = G0 / (np.linalg.norm(G0) + 1e-9) * F  # Final generation rate term.

                diff = best - X[i]  # Calculate the difference from the best solution.
                S = self.sigmoid(10 * diff / (np.linalg.norm(X[i]) + 1e-9)) if self.use_sigmoid else 1.0
                # Apply the sigmoid mapping for normalization.
                # Generate a new particle: the equilibrium term and global exploration term follow EO.
                # T is the normalized initial temperature decayed by the original search formula.
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

                X_check = self.trans_fur(X_new, self.mapping_table)
                # s_link=self.convert_virtual_links_to_substrate_links(X_check)
                # s_link_result=self.map_slinks_to_paths_batch(s_link)
                if self.has_duplicate_mappings(X_check):
                    # print(X_check)
                    # Discard invalid solutions and skip this iteration.
                    continue

                # 2. Discard blacklisted solutions immediately.
                # if tuple(X_check) in tabu_best_list:
                # continue
                # 3. Generate link requests first.
                s_link = self.convert_virtual_links_to_substrate_links(X_check)

                # 4. Perform link mapping.
                s_link_result = self.map_slinks_to_paths_numpy_test(s_link)
                if s_link_result is False:
                    continue

                # 5. Verify the mapped links are actually feasible.
                feasible, changed_edges = self.check_mapped_links_feasible(s_link_result)

                if not feasible:
                    continue
                found_feasible = True
                # Evaluate the new solution's fitness.
                fit_new = self._evaluate_fitness(
                    X_check,
                    s_link,
                    s_link_result,
                    virtual_degrees
                )

                # Greedily replace the previous solution if fitness improves, and update the global best if appropriate.
                if fit_new < fitness[i]:
                    X[i] = X_new
                    fitness[i] = fit_new
                    if fit_new < best_fit:
                        best = X_new.copy()
                        best_fit = fit_new
                        improved_this_iteration = True
                        last_improvement_iteration = t + 1
                        global_best_update_count += 1
                        time_to_final_best_seconds = time.perf_counter() - run_start_time
                        self.last_improvement_iteration = last_improvement_iteration
                        self.global_best_update_count = global_best_update_count
                        self.time_to_final_best_seconds = time_to_final_best_seconds
                        best_list = self.trans_fur(best, self.mapping_table)
                        prev_best_list = best_list
                        prev_best_slink = s_link
                        prev_best_slink_result = s_link_result
            # Append X to the history pool; this pool keeps growing, whereas the paper uses only two entries.
            position_history.append(X.copy())
            if self.use_implicit and t > self.iters * 0.3 and len(position_history) == 10:
                X += 0.2 * (self.implicit_update(list(position_history)) - X)
            # Is this the historical best?
            convergence_curve.append(np.log10(best_fit + 1e-10))
            iteration_records.append({
                "iteration": t + 1,
                "normalized_temperature": float(T0),
                "average_network_temperature": self.average_network_temperature,
                "update_temperature": float(T),
                "best_fitness": float(best_fit),
                "log10_best_fitness": float(np.log10(best_fit + 1e-10)),
                "improved": improved_this_iteration,
                "last_improvement_iteration": last_improvement_iteration,
                "global_best_update_count": global_best_update_count,
                "iteration_runtime_seconds": time.perf_counter() - iteration_start_time,
                "elapsed_seconds": time.perf_counter() - run_start_time
            })

        # End of the loop.
        total_runtime_seconds = time.perf_counter() - run_start_time
        self.last_total_iterations = self.iters
        self.last_runtime_seconds = total_runtime_seconds
        if not found_feasible:
            self.write_iteration_history({
                "optimizer": "EO2",
                "temperature_scale": self._temperature_scale,
                "average_network_temperature": self.average_network_temperature,
                "update_temperature": float(T),
                "vnr_id": self.vnr_id,
                "success": False,
                "total_iterations": self.iters,
                "final_fitness": None,
                "last_improvement_iteration": None,
                "global_best_update_count": 0,
                "time_to_final_best_seconds": None,
                "total_runtime_seconds": total_runtime_seconds,
                "iterations": iteration_records
            })
            if verbose:
                print("EO2 finished: no feasible mapping found.")
            return False, None, None, None

        self.write_iteration_history({
            "optimizer": "EO2",
                "temperature_scale": self._temperature_scale,
            "average_network_temperature": self.average_network_temperature,
            "update_temperature": float(T),
            "vnr_id": self.vnr_id,
            "success": True,
            "total_iterations": self.iters,
            "final_fitness": float(best_fit),
            "last_improvement_iteration": last_improvement_iteration,
            "global_best_update_count": global_best_update_count,
            "time_to_final_best_seconds": time_to_final_best_seconds,
            "total_runtime_seconds": total_runtime_seconds,
            "iterations": iteration_records
        })
        if verbose:
            print(
                f"EO2 finished: best_fitness={best_fit}, "
                f"average_network_temperature={self.average_network_temperature}, "
                f"last_improvement_iteration={last_improvement_iteration}, "
                f"time_to_final_best_seconds={time_to_final_best_seconds:.6f}, "
                f"total_runtime_seconds={total_runtime_seconds:.6f}"
            )
        return prev_best_list, best_fit, convergence_curve, prev_best_slink_result

    def run(self, verbose=True):
        return self.optimize(verbose=verbose)
