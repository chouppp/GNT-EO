import inspect
import json
import time

import numpy as np

from EO_test_two_stage_link import EO2_twostage


class GA_twostage(EO2_twostage):
    """GNT-GA baseline: tournament selection, uniform crossover and mutation."""

    def __init__(self, *args, crossover_prob=0.8, mutation_prob=None,
                 mutation_sigma=0.08, tournament_size=3, elite_count=1,
                 max_offspring_attempts=30, **kwargs):
        # Support both EO2 constructor variants used in the project.
        domain_temperatures = kwargs.pop("domain_temperatures", None)
        domains = kwargs.pop("domains", None)
        init_mode = kwargs.pop("init_mode", "temperature_guided")
        self.record_iteration_history = kwargs.pop("record_iteration_history", True)
        self.iteration_history_path = kwargs.pop(
            "iteration_history_path", "ga_iteration_history.jsonl"
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

        if not (0.0 <= crossover_prob <= 1.0):
            raise ValueError("crossover_prob must be in [0, 1].")
        if mutation_prob is None:
            mutation_prob = 1.0 / max(self.dim, 1)
        if not (0.0 <= mutation_prob <= 1.0):
            raise ValueError("mutation_prob must be in [0, 1].")
        if mutation_sigma < 0 or tournament_size < 2 or elite_count < 1:
            raise ValueError("Invalid GA hyperparameters.")

        self.domain_temperatures = domain_temperatures
        self.domains = domains
        self.init_mode = init_mode
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_sigma = mutation_sigma
        self.tournament_size = tournament_size
        self.elite_count = min(elite_count, self.pop_size)
        self.max_offspring_attempts = max_offspring_attempts

    def _evaluate_individual(self, x):
        """Apply the shared decode, VNE constraint, path, and fitness logic."""
        x = np.clip(np.asarray(x, dtype=float), self.lb, self.ub)
        decoded = self.trans_fur(x, self.mapping_table)
        if self.has_duplicate_mappings(decoded):
            return False, np.inf, None, None

        s_link = self.convert_virtual_links_to_substrate_links(decoded)
        s_link_result = self.map_slinks_to_paths_numpy_test(s_link)
        if s_link_result is False:
            return False, np.inf, None, None

        feasible, _ = self.check_mapped_links_feasible(s_link_result)
        if not feasible:
            return False, np.inf, None, None

        virtual_degrees = self.build_virtual_degrees(self.virtual_edges)
        fitness = self.func(
            s_link=s_link,
            substrate_nodes=self.substrate_nodes,
            substrate_edges=self.substrate_edges,
            virtual_degrees=virtual_degrees,
            s_link_result=s_link_result,
            domain_temperatures=self.domain_temperatures,
            domains=self.domains,
        )
        return True, float(fitness), decoded, s_link_result

    def _make_initial_population(self, initial_population=None):
        if initial_population is None:
            X, _, _, _ = self.generate_init_list()
            if X is False:
                return None
        else:
            X = np.asarray(initial_population, dtype=float).copy()
            if X.shape != (self.pop_size, self.dim):
                raise ValueError(
                    f"initial_population must have shape ({self.pop_size}, {self.dim})."
                )

        population = []
        for x in X:
            feasible, fitness, decoded, slink_result = self._evaluate_individual(x)
            if not feasible:
                if initial_population is not None:
                    raise ValueError("initial_population contains an infeasible individual.")
                continue
            population.append((x.copy(), fitness, decoded, slink_result))

        if not population:
            return None
        while len(population) < self.pop_size:
            x, fitness, decoded, slink_result = min(population, key=lambda item: item[1])
            population.append((x.copy(), fitness, decoded.copy(), list(slink_result)))
        return population[:self.pop_size]

    def _tournament_select(self, population):
        size = min(self.tournament_size, len(population))
        candidates = np.random.choice(len(population), size=size, replace=False)
        winner = min(candidates, key=lambda idx: population[idx][1])
        return population[winner][0].copy()

    def _uniform_crossover(self, parent1, parent2):
        mask = np.random.rand(self.dim) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2

    def _gaussian_mutation(self, child):
        child = child.copy()
        mask = np.random.rand(self.dim) < self.mutation_prob
        if np.any(mask):
            child[mask] += np.random.normal(0.0, self.mutation_sigma, np.sum(mask))
        return np.clip(child, self.lb, self.ub)

    def _save_iteration_history(self, final_fitness):
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

    def optimize(self, verbose=True, initial_population=None):
        start_time = time.perf_counter()
        population = self._make_initial_population(initial_population)
        if population is None:
            print("❌ 初始解生成失败，判定该虚拟网络不可映射")
            return False, None, None, None

        best = min(population, key=lambda item: item[1])
        best_x, best_fitness, best_mapping, best_slink_result = best
        best_x = best_x.copy()
        best_fitness = float(best_fitness)

        self.last_improvement_iteration = 0
        self.global_best_update_count = 0
        self.time_to_final_best_seconds = time.perf_counter() - start_time
        self.iteration_history = []
        convergence_curve = []

        for t in range(self.iters):
            iteration_start_time = time.perf_counter()
            improved_this_iteration = False
            new_population = []

            # Elitism: keep the best individuals unchanged.
            ranked_population = sorted(population, key=lambda item: item[1])
            for x, fitness, decoded, slink_result in ranked_population[:self.elite_count]:
                new_population.append((x.copy(), fitness, decoded.copy(), list(slink_result)))

            attempts = 0
            max_attempts = self.max_offspring_attempts * self.pop_size
            while len(new_population) < self.pop_size and attempts < max_attempts:
                attempts += 1
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)

                if np.random.rand() < self.crossover_prob:
                    child1, child2 = self._uniform_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2

                for child in (child1, child2):
                    child = self._gaussian_mutation(child)
                    feasible, fitness, decoded, slink_result = self._evaluate_individual(child)
                    if feasible:
                        new_population.append((child, fitness, decoded, slink_result))
                    if len(new_population) >= self.pop_size:
                        break

            # Preserve feasible parents if crossover/mutation yielded too few
            # feasible children. This avoids adding infeasible individuals.
            fill_index = 0
            while len(new_population) < self.pop_size:
                x, fitness, decoded, slink_result = ranked_population[
                    fill_index % len(ranked_population)
                ]
                new_population.append((x.copy(), fitness, decoded.copy(), list(slink_result)))
                fill_index += 1

            population = new_population[:self.pop_size]
            current_best = min(population, key=lambda item: item[1])
            if current_best[1] < best_fitness:
                best_x, best_fitness, best_mapping, best_slink_result = current_best
                best_x = best_x.copy()
                best_fitness = float(best_fitness)
                self.last_improvement_iteration = t + 1
                self.global_best_update_count += 1
                self.time_to_final_best_seconds = time.perf_counter() - start_time
                improved_this_iteration = True

            convergence_curve.append(np.log10(best_fitness + 1e-10))
            self.iteration_history.append({
                "iteration": t + 1,
                "best_fitness": float(best_fitness),
                "log10_best_fitness": float(convergence_curve[-1]),
                "improved": improved_this_iteration,
                "last_improvement_iteration": self.last_improvement_iteration,
                "global_best_update_count": self.global_best_update_count,
                "iteration_runtime_seconds": time.perf_counter() - iteration_start_time,
                "elapsed_seconds": time.perf_counter() - start_time,
            })

            if verbose and (t + 1) % 10 == 0:
                print(
                    f"GA Iter {t + 1}/{self.iters} - "
                    f"Best Fitness: {best_fitness} - best: {best_mapping}"
                )

        self.last_runtime_seconds = time.perf_counter() - start_time
        self.last_total_iterations = self.iters
        last_update = (
            "初始种群（第 0 次迭代）"
            if self.last_improvement_iteration == 0
            else f"第 {self.last_improvement_iteration} 次迭代"
        )
        print("\n========== GA Optimization Statistics ==========")
        print(f"总迭代次数: {self.last_total_iterations}")
        print(f"最终最优解最后更新于: {last_update}")
        print(f"全局最优解更新次数: {self.global_best_update_count}")
        print(f"找到最终最优解耗时: {self.time_to_final_best_seconds:.6f} s")
        print(f"算法总耗时: {self.last_runtime_seconds:.6f} s")
        print("================================================\n")
        self._save_iteration_history(best_fitness)
        return best_mapping, best_fitness, convergence_curve, best_slink_result

    def run(self, verbose=True, initial_population=None):
        return self.optimize(verbose=verbose, initial_population=initial_population)