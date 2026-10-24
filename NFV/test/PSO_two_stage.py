import inspect
import json
import time

import numpy as np

# This is the EO2 implementation imported by the user's current launcher.
from EO_test_two_stage_link import EO2_twostage


class PSO_twostage(EO2_twostage):
    """GNT-PSO baseline with standard velocity-position updates.

    The inherited methods retain exactly the same GNT-guided initialization,
    continuous interval decoding, link mapping, bandwidth checks, and fitness
    function as EO2.  Only the EO2 concentration update is replaced by PSO.
    """

    def __init__(self, *args, w_max=0.9, w_min=0.4, c1=1.5, c2=1.5,
                 v_max=0.20, infeasible_velocity_decay=0.5, **kwargs):
        # EO2 files in this project have two constructor variants: one accepts
        # the GNT arguments and one does not.  Inspect the inherited class so
        # this PSO baseline works with either version.
        domain_temperatures = kwargs.pop("domain_temperatures", None)
        domains = kwargs.pop("domains", None)
        init_mode = kwargs.pop("init_mode", "temperature_guided")
        self.record_iteration_history = kwargs.pop("record_iteration_history", True)
        self.iteration_history_path = kwargs.pop(
            "iteration_history_path", "pso_iteration_history.jsonl"
        )
        self.vnr_id = kwargs.pop("vnr_id", None)

        parent_parameters = inspect.signature(EO2_twostage.__init__).parameters
        parent_args = list(args)
        parent_kwargs = dict(kwargs)

        for name, value in (
            ("domain_temperatures", domain_temperatures),
            ("domains", domains),
            ("init_mode", init_mode),
        ):
            parameter = parent_parameters.get(name)
            if parameter is None:
                continue
            if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                parent_args.append(value)
            else:
                parent_kwargs[name] = value

        super().__init__(*parent_args, **parent_kwargs)

        self.domain_temperatures = domain_temperatures
        self.domains = domains
        self.init_mode = init_mode

        if not (0 < w_min <= w_max):
            raise ValueError("Require 0 < w_min <= w_max.")
        if c1 <= 0 or c2 <= 0 or v_max <= 0:
            raise ValueError("c1, c2, and v_max must be positive.")

        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.infeasible_velocity_decay = infeasible_velocity_decay

    def _save_iteration_history(self, final_fitness):
        """Append all iteration records of this VNR as one JSONL object."""
        if not self.record_iteration_history:
            return

        record = {
            "vnr_id": self.vnr_id,
            "total_iterations": self.last_total_iterations,
            "final_fitness": float(final_fitness),
            "last_improvement_iteration": self.last_improvement_iteration,
            "global_best_update_count": self.global_best_update_count,
            "time_to_final_best_seconds": self.time_to_final_best_seconds,
            "total_runtime_seconds": self.last_runtime_seconds,
            "iterations": self.iteration_history,
        }
        with open(self.iteration_history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _evaluate_particle(self, x):
        """Evaluate one decoded VNE solution under the shared constraints."""
        x = np.clip(np.asarray(x, dtype=float), self.lb, self.ub)
        decoded = self.trans_fur(x, self.mapping_table)

        if self.has_duplicate_mappings(decoded):
            return False, np.inf, None, None, None

        s_link = self.convert_virtual_links_to_substrate_links(decoded)
        s_link_result = self.map_slinks_to_paths_numpy_test(s_link)
        if s_link_result is False:
            return False, np.inf, None, None, None

        feasible, _ = self.check_mapped_links_feasible(s_link_result)
        if not feasible:
            return False, np.inf, None, None, None

        virtual_degrees = self.build_virtual_degrees(self.virtual_edges)
        fit = self.func(
            s_link=s_link,
            substrate_nodes=self.substrate_nodes,
            substrate_edges=self.substrate_edges,
            virtual_degrees=virtual_degrees,
            s_link_result=s_link_result,
            domain_temperatures=self.domain_temperatures,
            domains=self.domains,
        )
        return True, float(fit), decoded, s_link, s_link_result

    def _make_initial_population(self, initial_population=None):
        """Return a feasible population and its cached evaluation results."""
        if initial_population is None:
            X, decoded_list, slink_list, slink_result_list = self.generate_init_list()
            if X is False:
                return None

            fitness = np.empty(len(X), dtype=float)
            for i in range(len(X)):
                virtual_degrees = self.build_virtual_degrees(self.virtual_edges)
                fitness[i] = self.func(
                    s_link=slink_list[i],
                    substrate_nodes=self.substrate_nodes,
                    substrate_edges=self.substrate_edges,
                    virtual_degrees=virtual_degrees,
                    s_link_result=slink_result_list[i],
                    domain_temperatures=self.domain_temperatures,
                    domains=self.domains,
                )
            return X, fitness, decoded_list, slink_list, slink_result_list

        # This branch allows EO2, GA, and PSO to begin from the identical X0.
        X = np.asarray(initial_population, dtype=float).copy()
        if X.shape != (self.pop_size, self.dim):
            raise ValueError(
                "initial_population must have shape "
                f"({self.pop_size}, {self.dim}), got {X.shape}."
            )

        fitness = np.empty(self.pop_size, dtype=float)
        decoded_list, slink_list, slink_result_list = [], [], []
        for i in range(self.pop_size):
            feasible, fit, decoded, s_link, s_link_result = self._evaluate_particle(X[i])
            if not feasible:
                raise ValueError("initial_population contains an infeasible particle.")
            fitness[i] = fit
            decoded_list.append(decoded)
            slink_list.append(s_link)
            slink_result_list.append(s_link_result)
        return X, fitness, decoded_list, slink_list, slink_result_list

    def optimize(self, verbose=True, initial_population=None):
        start_time = time.perf_counter()
        initial = self._make_initial_population(initial_population)
        if initial is None:
            print("❌ 初始解生成失败，判定该虚拟网络不可映射")
            return False, None, None, None

        X, fitness, decoded_list, slink_list, slink_result_list = initial

        # Standard PSO state: each particle remembers its own best position.
        V = np.random.uniform(-self.v_max, self.v_max, size=(self.pop_size, self.dim))
        pbest = X.copy()
        pbest_fit = fitness.copy()

        best_idx = int(np.argmin(pbest_fit))
        gbest = pbest[best_idx].copy()
        gbest_fit = float(pbest_fit[best_idx])
        best_mapping = decoded_list[best_idx]
        best_slink_result = slink_result_list[best_idx]

        # Iteration 0 denotes the best solution already present in the
        # initial population. A positive value means PSO found the final best
        # solution during that iteration.
        self.last_improvement_iteration = 0
        self.global_best_update_count = 0
        self.time_to_final_best_seconds = time.perf_counter() - start_time
        self.iteration_history = []

        convergence_curve = []
        found_feasible = True  # generate_init_list() already guarantees feasibility.

        for t in range(self.iters):
            iteration_start_time = time.perf_counter()
            improved_this_iteration = False
            # Standard linearly decreasing inertia weight; this is not EO2 cooling.
            w = self.w_max - (self.w_max - self.w_min) * (t / max(self.iters - 1, 1))

            for i in range(self.pop_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                V[i] = (
                    w * V[i]
                    + self.c1 * r1 * (pbest[i] - X[i])
                    + self.c2 * r2 * (gbest - X[i])
                )
                V[i] = np.clip(V[i], -self.v_max, self.v_max)
                candidate = np.clip(X[i] + V[i], self.lb, self.ub)

                feasible, fit_new, decoded, _, s_link_result = self._evaluate_particle(candidate)
                if not feasible:
                    # Constraint handler: retain the last feasible position and
                    # reduce momentum instead of introducing a random restart.
                    V[i] *= self.infeasible_velocity_decay
                    continue

                found_feasible = True
                X[i] = candidate
                fitness[i] = fit_new

                if fit_new < pbest_fit[i]:
                    pbest[i] = candidate.copy()
                    pbest_fit[i] = fit_new

                    if fit_new < gbest_fit:
                        gbest = candidate.copy()
                        gbest_fit = float(fit_new)
                        best_mapping = decoded
                        best_slink_result = s_link_result
                        self.last_improvement_iteration = t + 1
                        self.global_best_update_count += 1
                        improved_this_iteration = True
                        self.time_to_final_best_seconds = (
                            time.perf_counter() - start_time
                        )

            convergence_curve.append(np.log10(gbest_fit + 1e-10))
            self.iteration_history.append({
                "iteration": t + 1,
                "best_fitness": float(gbest_fit),
                "log10_best_fitness": float(convergence_curve[-1]),
                "improved": improved_this_iteration,
                "last_improvement_iteration": self.last_improvement_iteration,
                "global_best_update_count": self.global_best_update_count,
                "iteration_runtime_seconds": time.perf_counter() - iteration_start_time,
                "elapsed_seconds": time.perf_counter() - start_time,
            })

            if verbose and (t + 1) % 10 == 0:
                print(
                    f"PSO Iter {t + 1}/{self.iters} - "
                    f"Best Fitness: {gbest_fit} - best: {best_mapping}"
                )

        self.last_runtime_seconds = time.perf_counter() - start_time
        self.last_total_iterations = self.iters
        if not found_feasible:
            print("❌ 未找到任何可行映射")
            return False, None, None, None

        if self.last_improvement_iteration == 0:
            improvement_text = "初始种群（第 0 次迭代）"
        else:
            improvement_text = f"第 {self.last_improvement_iteration} 次迭代"

        print("\n========== PSO Optimization Statistics ==========")
        print(f"总迭代次数: {self.last_total_iterations}")
        print(f"最终最优解最后更新于: {improvement_text}")
        print(f"全局最优解更新次数: {self.global_best_update_count}")
        print(f"找到最终最优解耗时: {self.time_to_final_best_seconds:.6f} s")
        print(f"算法总耗时: {self.last_runtime_seconds:.6f} s")
        print("=================================================\n")
        self._save_iteration_history(gbest_fit)
        return best_mapping, gbest_fit, convergence_curve, best_slink_result

    def run(self, verbose=True, initial_population=None):
        return self.optimize(verbose=verbose, initial_population=initial_population)