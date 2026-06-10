from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

import conservation_experiment_config as cfg


@dataclass
class RoundResult:
    algorithm: str
    area: int
    round_index: int
    chosen_arm: int
    chosen_period: int
    reward: float
    cumulative_reward: float
    average_reward: float


@dataclass
class SummaryResult:
    algorithm: str
    area: int
    rounds: int
    cumulative_reward: float
    average_reward: float
    arm_counts: str
    arm_scores: str


def load_dataset(data_path, nrows=cfg.DATA_NROWS):
    return pd.read_csv(data_path, header=None, nrows=nrows)


def load_area_data(df, area_index, task_type_num):
    start = area_index * task_type_num
    end = start + task_type_num
    return df.iloc[start:end].to_numpy()


def reservation_energy(period, reserved_resource, zeta_value=cfg.ZETA):
    return zeta_value * reserved_resource / period


def aggregate_resource_demand(task_sizes, time_index):
    return np.sum(task_sizes[:, time_index])


def compute_reserved_resource(task_sizes, time_index, xi_max_value):
    current_demand = aggregate_resource_demand(task_sizes, time_index)
    previous_index = max(time_index - 1, 0)
    previous_demand = aggregate_resource_demand(task_sizes, previous_index)
    load_ratio = np.clip(current_demand / max(xi_max_value, cfg.EPS), 0, 1)
    load_adjustment = np.log2(2 - load_ratio)
    demand_variation = abs(current_demand - previous_demand)
    return current_demand + load_adjustment * demand_variation


def aggregated_demand_reward_function(
    period,
    task_sizes,
    t,
    xi_max_value,
    zeta_value=cfg.ZETA,
):
    reservation_error = 0
    energy_cost = 0
    for offset in range(period):
        current_time = t + offset
        actual_time = t + offset + 1
        psi_n_t = compute_reserved_resource(task_sizes, current_time, xi_max_value)
        actual_demand = aggregate_resource_demand(task_sizes, actual_time)
        reservation_error += abs(psi_n_t - actual_demand)
        energy_cost += reservation_energy(period, psi_n_t, zeta_value)
    upsilon = -reservation_error - energy_cost
    return upsilon / period


def reward_function(
    period,
    task_sizes,
    t,
    xi_max_value,
    zeta_value=cfg.ZETA,
    use_aggregated_demand_reward=cfg.USE_AGGREGATED_DEMAND_REWARD,
):
    if use_aggregated_demand_reward:
        return aggregated_demand_reward_function(
            period,
            task_sizes,
            t,
            xi_max_value,
            zeta_value,
        )

    reservation_error_step = 0
    reservation_error = 0
    reserved_resource = cfg.RESERVED_RESOURCE
    for task_index in range(task_sizes.shape[0]):
        for offset in range(period):
            if t == 0:
                psi_n_t = task_sizes[task_index][t + offset] + 0
            else:
                psi_n_t = (
                    task_sizes[task_index][t]
                    + task_sizes[task_index][t + offset]
                    - task_sizes[task_index][t - 1 + offset]
                )

            actual_demand = task_sizes[task_index][t + 1 + offset]
            if actual_demand > psi_n_t:
                reservation_error_step += (
                    cfg.FACTOR * cfg.REWARD_SCALE * (actual_demand - psi_n_t)
                )
            else:
                reservation_error_step -= (
                    cfg.FACTOR * cfg.REWARD_SCALE * (actual_demand - psi_n_t)
                )
            reservation_error += reservation_error_step

    phi = reservation_energy(period, reserved_resource, zeta_value)
    upsilon = -reservation_error - phi
    return upsilon / period


class UCBAlgorithm:
    name = "UCB"

    def __init__(
        self,
        arms,
        c=cfg.UCB_C,
        use_normalized_confidence=cfg.USE_NORMALIZED_CONFIDENCE,
        use_cumulative_reward=cfg.USE_CUMULATIVE_REWARD,
    ):
        self.arms = list(arms)
        self.num_arms = len(arms)
        self.c = c
        self.n = np.zeros(self.num_arms)
        self.s = np.zeros(self.num_arms)
        self.t = 0
        self.use_normalized_confidence = use_normalized_confidence
        self.use_cumulative_reward = use_cumulative_reward

    def confidence_reward(self, arm_index):
        mean_reward = self.s[arm_index] / self.n[arm_index]
        if self.use_normalized_confidence:
            confidence = self.c * np.sqrt(
                ((np.log(self.t) + 1) * np.log(2)) / (2 * self.n[arm_index])
            )
        else:
            confidence = self.c * np.sqrt(np.log(self.t) / self.n[arm_index])
        return mean_reward + confidence

    def select_arm(self):
        self.t += 1
        ucb_values = []
        for arm_index in range(self.num_arms):
            if self.n[arm_index] == 0:
                return arm_index
            ucb_values.append(self.confidence_reward(arm_index))
        return int(np.argmax(ucb_values))

    def update(self, chosen_arm, reward):
        self.n[chosen_arm] += 1
        if self.use_cumulative_reward:
            self.s[chosen_arm] += reward
        else:
            self.s[chosen_arm] = (
                cfg.UPDATE_CURRENT_WEIGHT * reward
                + cfg.UPDATE_HISTORY_WEIGHT * self.s[chosen_arm]
            )


class EpsilonGreedyMAB:
    name = "MAB(e)"

    def __init__(self, arms, epsilon=cfg.MAB_EPSILON):
        self.arms = list(arms)
        self.num_arms = len(arms)
        self.epsilon = epsilon
        self.n = np.zeros(self.num_arms)
        self.s = np.zeros(self.num_arms)

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(self.num_arms))
        mean_rewards = self.s / (self.n + cfg.MAB_MEAN_DENOMINATOR_EPS)
        return int(np.argmax(mean_rewards))

    def update(self, chosen_arm, reward):
        self.n[chosen_arm] += 1
        self.s[chosen_arm] = (
            cfg.UPDATE_CURRENT_WEIGHT * reward
            + cfg.UPDATE_HISTORY_WEIGHT * self.s[chosen_arm]
        )


class RanchAlgorithm:
    name = "Ranch"

    def __init__(self, arms):
        self.arms = list(arms)
        self.num_arms = len(arms)
        self.n = np.zeros(self.num_arms)
        self.s = np.zeros(self.num_arms)

    def select_arm(self):
        return int(np.random.choice(self.num_arms))

    def update(self, chosen_arm, reward):
        self.n[chosen_arm] += 1
        self.s[chosen_arm] = (
            cfg.UPDATE_CURRENT_WEIGHT * reward
            + cfg.UPDATE_HISTORY_WEIGHT * self.s[chosen_arm]
        )


class ThompsonSampling:
    name = "TS"

    def __init__(self, arms):
        self.arms = list(arms)
        self.num_arms = len(arms)
        self.alpha = np.full(self.num_arms, cfg.TS_ALPHA_INITIAL)
        self.beta = np.full(self.num_arms, cfg.TS_BETA_INITIAL)
        self.n = np.zeros(self.num_arms)
        self.s = np.zeros(self.num_arms)

    def select_arm(self):
        sampled_values = [
            np.random.beta(self.alpha[arm_index], self.beta[arm_index])
            for arm_index in range(self.num_arms)
        ]
        return int(np.argmax(sampled_values))

    def update(self, chosen_arm, reward):
        self.n[chosen_arm] += 1
        self.s[chosen_arm] = (
            cfg.UPDATE_CURRENT_WEIGHT * reward
            + cfg.UPDATE_HISTORY_WEIGHT * self.s[chosen_arm]
        )
        if reward > 0:
            self.alpha[chosen_arm] = (
                cfg.UPDATE_CURRENT_WEIGHT * reward
                + cfg.UPDATE_HISTORY_WEIGHT * self.alpha[chosen_arm]
            )
        else:
            self.beta[chosen_arm] = (
                -cfg.UPDATE_CURRENT_WEIGHT * reward
                + cfg.UPDATE_HISTORY_WEIGHT * self.alpha[chosen_arm]
            )


def format_array(values):
    return "[" + ", ".join(f"{float(value):.6f}" for value in values) + "]"


def run_algorithm(
    algorithm,
    area,
    task_sizes,
    rounds,
    use_aggregated_demand_reward=False,
):
    xi_max_value = np.max(np.sum(task_sizes, axis=0))
    max_rounds = task_sizes.shape[1] - max(algorithm.arms) - 1
    actual_rounds = min(rounds, max_rounds)
    if actual_rounds <= 0:
        raise ValueError("The dataset does not contain enough time slots.")

    cumulative_reward = 0
    round_results = []
    for t in range(actual_rounds):
        chosen_arm = algorithm.select_arm()
        chosen_period = algorithm.arms[chosen_arm]
        reward = reward_function(
            chosen_period,
            task_sizes,
            t,
            xi_max_value,
            use_aggregated_demand_reward=use_aggregated_demand_reward,
        )
        algorithm.update(chosen_arm, reward)
        cumulative_reward += reward
        average_reward = cumulative_reward / (t + 1)
        round_results.append(
            RoundResult(
                algorithm=algorithm.name,
                area=area,
                round_index=t,
                chosen_arm=chosen_arm,
                chosen_period=chosen_period,
                reward=float(reward),
                cumulative_reward=float(cumulative_reward),
                average_reward=float(average_reward),
            )
        )

    summary = SummaryResult(
        algorithm=algorithm.name,
        area=area,
        rounds=actual_rounds,
        cumulative_reward=float(cumulative_reward),
        average_reward=float(cumulative_reward / actual_rounds),
        arm_counts=format_array(algorithm.n),
        arm_scores=format_array(algorithm.s),
    )
    return round_results, summary


def build_algorithms(arms=cfg.ARMS, ucb_c=cfg.UCB_C, epsilon=cfg.MAB_EPSILON):
    return [
        UCBAlgorithm(arms, c=ucb_c),
        EpsilonGreedyMAB(arms, epsilon=epsilon),
        RanchAlgorithm(arms),
        ThompsonSampling(arms),
    ]


def main():
    df = load_dataset(cfg.DATA_PATH, cfg.DATA_NROWS)
    all_round_results = []
    all_summary_results = []

    for area in range(cfg.AREA_NUM):
        task_sizes = load_area_data(df, area, cfg.TASK_TYPE_NUM)
        for algorithm in build_algorithms(cfg.ARMS, cfg.UCB_C, cfg.MAB_EPSILON):
            round_results, summary = run_algorithm(
                algorithm,
                area,
                task_sizes,
                cfg.ROUNDS,
                use_aggregated_demand_reward=cfg.USE_AGGREGATED_DEMAND_REWARD,
            )
            all_round_results.extend(round_results)
            all_summary_results.append(summary)
            print(
                f"{summary.algorithm}, area={summary.area}, "
                f"average_reward={summary.average_reward:.6f}, "
                f"cumulative_reward={summary.cumulative_reward:.6f}"
            )

    pd.DataFrame(asdict(result) for result in all_round_results).to_csv(
        cfg.ROUND_OUTPUT,
        index=False,
    )
    pd.DataFrame(asdict(result) for result in all_summary_results).to_csv(
        cfg.SUMMARY_OUTPUT,
        index=False,
    )


if __name__ == "__main__":
    main()
