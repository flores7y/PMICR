import argparse
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
except ModuleNotFoundError:
    minimize = None
from offloading_common import init_link_rates, random_computation_capacity
import offloading_experiment_config as cfg


DEFAULT_TASK_SIZE = 10.0
DEFAULT_LOCAL_CAPACITY = 3.0
MAX_PRICE_SV = 3.0
PRICE_FACTOR = 1.0
COMPUTATION_COEFFICIENT = 1.0
DELAY_UTILITY_WEIGHT = 1.0
ENERGY_FACTOR = 0.01
PAYMENT_WEIGHT = 0.1
ENERGY_WEIGHT = 0.1


@dataclass
class OffloadingScenario:
    sample_index: int
    task: float
    local_capacity: float
    sv_capacity: float
    es_capacity: float
    rate_sv: float
    rate_es: float


@dataclass
class OffloadingResult:
    algorithm: str
    sample_index: int
    price_sv: float
    price_es: float
    offloading_ratio: float
    utility_tv: float
    utility_sv: float
    utility_es: float
    provider_utility: float
    total_utility: float
    rate_sv: float
    rate_es: float
    sv_capacity: float
    es_capacity: float


def build_scenario(
    sample_index=0,
    task=DEFAULT_TASK_SIZE,
    local_capacity=DEFAULT_LOCAL_CAPACITY,
):
    rate_sv, rate_es = init_link_rates(sample_index)
    sv_capacity, es_capacity = random_computation_capacity()
    return OffloadingScenario(
        sample_index=int(sample_index),
        task=float(task),
        local_capacity=float(local_capacity),
        sv_capacity=float(sv_capacity),
        es_capacity=float(es_capacity),
        rate_sv=float(rate_sv),
        rate_es=float(rate_es),
    )


class OffloadingSolverCore:
    algorithm = "Core"

    def __init__(self, scenario):
        self.scenario = scenario
        self.task = scenario.task
        self.price_m = cfg.INITIAL_PRICE_SV
        self.price_m_step = cfg.INITIAL_PRICE_STEP_SV
        self.price_n = cfg.INITIAL_PRICE_ES
        self.price_n_step = cfg.INITIAL_PRICE_STEP_ES
        self.price_factor = PRICE_FACTOR
        self.f = scenario.local_capacity
        self.f1 = scenario.sv_capacity
        self.fe1 = scenario.es_capacity
        self.rate_m = scenario.rate_sv
        self.rate_n = scenario.rate_es

        self.cal_per = COMPUTATION_COEFFICIENT
        self.alpha = DELAY_UTILITY_WEIGHT
        self.energy = ENERGY_FACTOR
        self.v = PAYMENT_WEIGHT
        self.eps = ENERGY_WEIGHT
        self.step = cfg.PRICE_UPDATE_STEP

        self.offloading = cfg.INITIAL_OFFLOADING_RATIO
        self.t_l = self.cal_per * self.task / self.f
        self.t_m = 0.0
        self.t_n = 0.0
        self.u_l = 0.0
        self.u_m = 0.0
        self.u_n = 0.0
        self.u_m_max = 0.0
        self.u_n_max = 0.0

    def _regularized_denominator(self, value):
        if abs(value) < cfg.EPS:
            return cfg.EPS if value >= 0 else -cfg.EPS
        return value

    def project_offloading(self, offloading, lower_bound):
        if offloading > 1:
            return 1 - lower_bound
        if offloading < lower_bound:
            return lower_bound
        return offloading

    def energy_con(self, capacity, ratio):
        return self.energy * self.cal_per * capacity * ratio * self.task

    def update_delays(self):
        self.t_m = (
            self.offloading * self.task / self.rate_m
            + self.cal_per * self.offloading * self.task / self.f1
        )
        self.t_n = (
            (1 - self.offloading) * self.task / self.rate_n
            + self.cal_per * (1 - self.offloading) * self.task / self.fe1
        )

    def cal_u_l(self):
        energy_penalty = self.eps * (
            self.energy_con(self.f1, self.offloading)
            + self.energy_con(self.fe1, 1 - self.offloading)
        )
        self.u_l = (
            self.alpha * math.log2(1 + self.t_l - max(self.t_m, self.t_n))
            - self.v * self.price_m_step * self.offloading * self.task
            - self.v * self.price_n_step * (1 - self.offloading) * self.task
            - energy_penalty
        )
        return self.u_l

    def result(self):
        self.update_delays()
        u_l = self.cal_u_l()
        u_m = self.cal_u_m()
        u_n = self.cal_u_n()
        provider_utility = u_m + u_n
        return OffloadingResult(
            algorithm=self.algorithm,
            sample_index=self.scenario.sample_index,
            price_sv=float(self.price_m),
            price_es=float(self.price_n),
            offloading_ratio=float(self.offloading),
            utility_tv=float(u_l),
            utility_sv=float(u_m),
            utility_es=float(u_n),
            provider_utility=float(provider_utility),
            total_utility=float(provider_utility + u_l),
            rate_sv=self.scenario.rate_sv,
            rate_es=self.scenario.rate_es,
            sv_capacity=self.scenario.sv_capacity,
            es_capacity=self.scenario.es_capacity,
        )


class PMICRSolver(OffloadingSolverCore):
    algorithm = "PMICR"

    def offloading_cal(self, price_m, price_n):
        self.update_delays()
        denominator = self._regularized_denominator(price_m - price_n)
        offloading = (
            (
                cfg.OFFLOADING_FORMULA_WEIGHT
                * (
                    1 / self.rate_m
                    + cfg.COMPUTATION_DELAY_SCALE * self.cal_per / self.f1
                )
                / denominator
                + self.cal_per * self.task / self.f
                + 1
            )
            / (
                self.task
                * (
                    cfg.COMPUTATION_DELAY_SCALE * self.cal_per / self.f1
                    + 1 / self.rate_m
                )
            )
        )
        lower_bound = cfg.OFFLOADING_LOWER_BASE + random.uniform(
            *cfg.OFFLOADING_LOWER_RANDOM_RANGE
        )
        return self.project_offloading(offloading, lower_bound)

    def cal_u_m(self):
        self.u_m = (
            self.price_m_step
            * self.task
            * np.exp(-self.price_factor * self.price_m)
            - self.energy
            * (
                self.f1
                * self.cal_per
                * self.offloading
                * self.task
                * np.exp(-self.price_factor * self.price_m)
            )
        )
        return self.u_m

    def cal_u_n(self):
        self.u_n = (
            self.price_n_step
            * self.task
            * np.exp(-self.price_factor * self.price_n)
            - self.energy
            * (
                self.fe1
                * self.cal_per
                * (1 - self.offloading)
                * self.task
                * np.exp(-self.price_factor * self.price_n)
            )
        )
        return self.u_n

    def cal_pm(self):
        self.update_delays()
        self.price_m_step -= cfg.PRICE_DECREASE_STEP
        self.u_m_max = self.cal_u_m()
        for _ in range(cfg.INNER_ITERATIONS):
            self.price_m_step = (
                self.price_m_step
                + (cfg.PRICE_UPDATE_TARGET - self.price_m_step)
                / cfg.PRICE_UPDATE_DIVISOR
                * self.step
            )
            upper = self.energy * self.f1 + cfg.PROVIDER_PRICE_UPPER_OFFSET / self.price_factor
            if self.price_m_step > upper:
                self.price_m_step = upper
            offloading = self.offloading_cal(self.price_m_step, self.price_n)
            if abs(self.offloading - offloading) > cfg.OFFLOADING_ADJUST_THRESHOLD:
                self.offloading += (
                    cfg.OFFLOADING_ADJUST_STEP
                    if self.offloading < offloading
                    else -cfg.OFFLOADING_ADJUST_STEP
                )
            else:
                self.offloading = offloading
            self.offloading = min(
                max(self.offloading, cfg.OFFLOADING_PROJECT_MIN),
                cfg.OFFLOADING_PROJECT_MAX,
            )
            self.u_m = self.cal_u_m()
            self.u_l = self.cal_u_l()
            if self.u_m > self.u_m_max:
                self.price_m = self.price_m_step
                self.u_m_max = self.u_m
                break
        return self.price_m, self.offloading

    def cal_pn(self):
        self.update_delays()
        self.price_n_step -= cfg.PRICE_DECREASE_STEP
        self.u_n_max = self.cal_u_n()
        for _ in range(cfg.INNER_ITERATIONS):
            self.price_n_step = (
                self.price_n_step
                + (cfg.PRICE_UPDATE_TARGET - self.price_n_step)
                / cfg.PRICE_UPDATE_DIVISOR
                * self.step
                * cfg.ES_PRICE_UPDATE_FACTOR
            )
            upper = self.energy * self.fe1 + cfg.PROVIDER_PRICE_UPPER_OFFSET / self.price_factor
            if self.price_n_step > upper:
                self.price_n_step = upper
            self.offloading = self.offloading_cal(self.price_m, self.price_n_step)
            self.offloading = min(
                max(self.offloading, cfg.OFFLOADING_PROJECT_MIN),
                cfg.OFFLOADING_PROJECT_MAX,
            )
            self.u_n = self.cal_u_n()
            self.u_l = self.cal_u_l()
            if self.u_n >= self.u_n_max:
                self.price_n = self.price_n_step
                self.u_n_max = self.u_n
                break
        return self.price_n, self.offloading

    def run(self, iterations=cfg.PMICR_ITERATIONS):
        self.offloading = self.offloading_cal(self.price_m, self.price_n)
        self.u_m_max = self.cal_u_m()
        self.u_n_max = self.cal_u_n()
        for _ in range(iterations):
            self.cal_pm()
            self.cal_pn()
        return self.result()


class WithoutRestrainSolver(PMICRSolver):
    algorithm = "WithoutRestrain"

    def cal_u_m(self):
        self.u_m = self.price_m_step * self.task - self.energy * (
            self.f1 * self.cal_per * self.offloading * self.task
        )
        return self.u_m

    def cal_u_n(self):
        self.u_n = self.price_n_step * self.task - self.energy * (
            self.fe1 * self.cal_per * (1 - self.offloading) * self.task
        )
        return self.u_n

    def cal_pn(self):
        self.update_delays()
        self.price_n_step -= cfg.PRICE_DECREASE_STEP
        self.u_n_max = self.cal_u_n()
        for _ in range(cfg.INNER_ITERATIONS):
            self.price_n_step = (
                self.price_n_step
                + (cfg.PRICE_UPDATE_TARGET - self.price_n_step)
                / cfg.PRICE_UPDATE_DIVISOR
                * self.step
                * cfg.ES_PRICE_UPDATE_FACTOR
                * cfg.VARIANT_ES_PRICE_UPDATE_FACTOR
            )
            upper = self.energy * self.fe1 + cfg.PROVIDER_PRICE_UPPER_OFFSET / self.price_factor
            if self.price_n_step > upper:
                self.price_n_step = upper
            self.offloading = self.offloading_cal(self.price_m, self.price_n_step)
            self.offloading = min(
                max(self.offloading, cfg.OFFLOADING_PROJECT_MIN),
                cfg.OFFLOADING_PROJECT_MAX,
            )
            self.u_n = self.cal_u_n()
            self.u_l = self.cal_u_l()
            if self.u_n >= self.u_n_max:
                self.price_n = self.price_n_step
                self.u_n_max = self.u_n
                break
        return self.price_n, self.offloading


class SimulatedAnnealingSolver(OffloadingSolverCore):
    algorithm = "SA"

    def __init__(
        self,
        scenario,
        initial_temp=cfg.SA_INITIAL_TEMP,
        cooling_rate=cfg.SA_COOLING_RATE,
        stop_temp=cfg.SA_STOP_TEMP,
    ):
        super().__init__(scenario)
        self.temp = initial_temp
        self.cooling_rate = cooling_rate
        self.stop_temp = stop_temp

    def offloading_cal(self, price_m, price_n):
        self.update_delays()
        denominator = self._regularized_denominator(price_m - price_n)
        offloading = (
            (
                cfg.OFFLOADING_FORMULA_WEIGHT
                * (
                    1 / self.rate_m
                    + cfg.COMPUTATION_DELAY_SCALE * self.cal_per / self.f1
                )
                / denominator
                + self.cal_per * self.task / self.f
                + 1
            )
            / (
                self.task
                * (
                    cfg.COMPUTATION_DELAY_SCALE * self.cal_per / self.f1
                    + 1 / self.rate_m
                )
            )
        )
        lower_bound = cfg.OFFLOADING_LOWER_BASE + random.uniform(
            *cfg.OFFLOADING_LOWER_RANDOM_RANGE
        )
        return self.project_offloading(offloading, lower_bound)

    def cal_u_m(self, price_m=None):
        price_m = self.price_m if price_m is None else price_m
        return (
            price_m * self.task * np.exp(-self.price_factor * price_m)
            - self.energy
            * (
                self.f1
                * self.cal_per
                * self.offloading
                * self.task
                * np.exp(-self.price_factor * price_m)
            )
        )

    def cal_u_n(self, price_n=None):
        price_n = self.price_n if price_n is None else price_n
        return (
            price_n * self.task * np.exp(-self.price_factor * price_n)
            - self.energy
            * (
                self.fe1
                * self.cal_per
                * (1 - self.offloading)
                * self.task
                * np.exp(-self.price_factor * price_n)
            )
        )

    def cal_pm(self):
        self.update_delays()
        current_best = self.cal_u_m(self.price_m)
        for _ in range(cfg.INNER_ITERATIONS):
            candidate = max(
                cfg.SA_MIN_PRICE_SV,
                min(
                    self.price_m + random.uniform(*cfg.SA_CANDIDATE_STEP_RANGE),
                    cfg.SA_MAX_PRICE_SV,
                ),
            )
            upper = (
                self.energy * self.f1 * self.offloading
                + cfg.PROVIDER_PRICE_UPPER_OFFSET / self.price_factor
            )
            candidate = min(candidate, upper)
            candidate_offloading = self.offloading_cal(candidate, self.price_n)
            old_price, old_offloading = self.price_m, self.offloading
            self.price_m, self.offloading = candidate, candidate_offloading
            candidate_utility = self.cal_u_m(candidate)
            if self.accept(current_best, candidate_utility):
                current_best = candidate_utility
            else:
                self.price_m, self.offloading = old_price, old_offloading
            self.price_m_step = self.price_m
            self.u_m = current_best
        return self.price_m, self.offloading

    def cal_pn(self):
        self.update_delays()
        current_best = self.cal_u_n(self.price_n)
        for _ in range(cfg.INNER_ITERATIONS):
            candidate = max(
                cfg.SA_MIN_PRICE_ES,
                min(
                    self.price_n + random.uniform(*cfg.SA_CANDIDATE_STEP_RANGE),
                    cfg.SA_MAX_PRICE_ES,
                ),
            )
            if candidate > self.price_m:
                candidate = self.price_n
            upper = (
                self.energy * self.fe1 * self.offloading
                + cfg.PROVIDER_PRICE_UPPER_OFFSET / self.price_factor
            )
            candidate = min(candidate, upper)
            candidate_offloading = self.offloading_cal(self.price_m, candidate)
            old_price, old_offloading = self.price_n, self.offloading
            self.price_n, self.offloading = candidate, candidate_offloading
            candidate_utility = self.cal_u_n(candidate)
            if self.accept(current_best, candidate_utility):
                current_best = candidate_utility
            else:
                self.price_n, self.offloading = old_price, old_offloading
            self.price_n_step = self.price_n
            self.u_n = current_best
        return self.price_n, self.offloading

    def accept(self, old_utility, new_utility):
        if new_utility > old_utility:
            return True
        probability = math.exp((new_utility - old_utility) / max(self.temp, cfg.EPS))
        return random.random() < probability

    def run(self):
        while self.temp > self.stop_temp:
            self.cal_pm()
            self.cal_pn()
            self.temp *= self.cooling_rate
        self.price_m_step = self.price_m
        self.price_n_step = self.price_n
        return self.result()


class LBFGSBSolver(OffloadingSolverCore):
    algorithm = "L-BFGS-B"

    def __init__(
        self,
        scenario,
        max_price_sv=MAX_PRICE_SV,
        max_price_es=cfg.LBFGSB_MAX_PRICE_ES,
    ):
        super().__init__(scenario)
        self.max_price_sv = max_price_sv
        self.max_price_es = max_price_es

    def offloading_cal(self, price_m, price_n):
        denominator = self._regularized_denominator(price_m - price_n)
        offloading = (
            (
                cfg.LBFGSB_OFFLOADING_FORMULA_WEIGHT
                * (
                    1 / self.rate_m
                    + cfg.COMPUTATION_DELAY_SCALE * self.cal_per / self.f1
                )
                / denominator
                + self.cal_per * self.task / self.f
                + 1
            )
            / (
                self.task
                * (
                    cfg.COMPUTATION_DELAY_SCALE * self.cal_per / self.f1
                    + 1 / self.rate_m
                )
            )
        )
        lower_bound = cfg.OFFLOADING_LOWER_BASE + random.uniform(
            *cfg.OFFLOADING_LOWER_RANDOM_RANGE
        )
        return max(lower_bound, min(offloading, 1 - lower_bound))

    def utility(self, prices):
        price_m, price_n = prices
        self.offloading = self.offloading_cal(price_m, price_n)
        self.price_m = price_m
        self.price_n = price_n
        self.price_m_step = price_m
        self.price_n_step = price_n
        self.update_delays()
        return -(self.cal_u_m() + self.cal_u_n())

    def cal_u_m(self):
        self.u_m = (
            self.price_m
            * self.offloading
            * self.task
            * np.exp(-self.price_factor * self.price_m)
            - self.energy
            * (
                self.f1
                * self.cal_per
                * self.offloading
                * self.task
                * np.exp(-self.price_factor * self.price_m)
            )
        )
        return self.u_m

    def cal_u_n(self):
        self.u_n = (
            self.price_n
            * (1 - self.offloading)
            * self.task
            * np.exp(-self.price_factor * self.price_n)
            - self.energy
            * (
                self.fe1
                * self.cal_per
                * (1 - self.offloading)
                * self.task
                * np.exp(-self.price_factor * self.price_n)
            )
        )
        return self.u_n

    def run(self):
        initial_prices = [
            random.uniform(cfg.INITIAL_PRICE_SV, self.max_price_sv),
            random.uniform(cfg.INITIAL_PRICE_ES, self.max_price_es),
        ]
        if minimize is None:
            best_prices = self.local_box_search(initial_prices)
            self.utility(best_prices)
            return self.result()

        result = minimize(
            self.utility,
            initial_prices,
            bounds=[
                (cfg.LBFGSB_BOUND_LOWER, self.max_price_sv),
                (cfg.LBFGSB_BOUND_LOWER, self.max_price_es),
            ],
            method="L-BFGS-B",
            options={"ftol": cfg.LBFGSB_FTOL, "maxiter": cfg.LBFGSB_MAXITER},
        )
        if not result.success:
            raise RuntimeError(f"L-BFGS-B failed: {result.message}")
        self.utility(result.x)
        return self.result()

    def local_box_search(self, initial_prices):
        best = np.array(initial_prices, dtype=float)
        best[0] = np.clip(best[0], cfg.LBFGSB_BOUND_LOWER, self.max_price_sv)
        best[1] = np.clip(best[1], cfg.LBFGSB_BOUND_LOWER, self.max_price_es)
        best_value = self.utility(best)
        step_sizes = np.array(cfg.LOCAL_BOX_INITIAL_STEP, dtype=float)

        while np.max(step_sizes) > cfg.LOCAL_BOX_STOP_STEP:
            improved = False
            for dim in range(2):
                for direction in (-1, 1):
                    candidate = best.copy()
                    candidate[dim] += direction * step_sizes[dim]
                    candidate[0] = np.clip(
                        candidate[0],
                        cfg.LBFGSB_BOUND_LOWER,
                        self.max_price_sv,
                    )
                    candidate[1] = np.clip(
                        candidate[1],
                        cfg.LBFGSB_BOUND_LOWER,
                        self.max_price_es,
                    )
                    value = self.utility(candidate)
                    if value < best_value:
                        best = candidate
                        best_value = value
                        improved = True
            if not improved:
                step_sizes *= cfg.LOCAL_BOX_DECAY
        return best


def run_all_algorithms(sample_index):
    scenario = build_scenario(sample_index)
    solvers = [
        PMICRSolver(scenario),
        LBFGSBSolver(scenario),
        SimulatedAnnealingSolver(scenario),
        WithoutRestrainSolver(scenario),
    ]
    return [solver.run() for solver in solvers]


def main():
    parser = argparse.ArgumentParser(
        description="Run offloading algorithms on shared samples."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=cfg.DEFAULT_SAMPLES,
        help="Number of sample indices to run.",
    )
    parser.add_argument(
        "--output",
        default=cfg.DEFAULT_OUTPUT,
        help="CSV file used to save comparison results.",
    )
    args = parser.parse_args()

    rows = []
    for sample_index in range(args.samples):
        for result in run_all_algorithms(sample_index):
            rows.append(asdict(result))

    data = pd.DataFrame(rows)
    output_path = Path(args.output)
    data.to_csv(output_path, index=False)
    print(data.to_string(index=False))
    print(f"\nSaved results to {output_path.resolve()}")


if __name__ == "__main__":
    main()
