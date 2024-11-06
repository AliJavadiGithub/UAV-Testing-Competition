import random
import math
from typing import List, Dict, Tuple
from aerialist.px4.drone_test import DroneTest
from aerialist.px4.obstacle import Obstacle
from testcase import TestCase


class ObstacleGenerator:
    """Handles the creation and transformation of obstacles."""

    min_size = Obstacle.Size(2, 2, 15)
    max_size = Obstacle.Size(20, 20, 25)
    min_position = Obstacle.Position(-40, 10, 0, 0)
    max_position = Obstacle.Position(30, 40, 0, 90)

    @staticmethod
    def create_random_obstacle() -> Obstacle:
        size = Obstacle.Size(
            l=random.uniform(ObstacleGenerator.min_size.l, ObstacleGenerator.max_size.l),
            w=random.uniform(ObstacleGenerator.min_size.w, ObstacleGenerator.max_size.w),
            h=random.uniform(ObstacleGenerator.min_size.h, ObstacleGenerator.max_size.h),
        )
        position = Obstacle.Position(
            x=random.uniform(ObstacleGenerator.min_position.x, ObstacleGenerator.max_position.x),
            y=random.uniform(ObstacleGenerator.min_position.y, ObstacleGenerator.max_position.y),
            z=0,
            r=random.uniform(ObstacleGenerator.min_position.r, ObstacleGenerator.max_position.r),
        )
        return Obstacle(size, position)

    @staticmethod
    def apply_action(action: str, obstacles: List[Obstacle]) -> List[Obstacle]:
        new_obstacles = []
        for obs in obstacles:
            if action == 'adjust_x':
                new_x = random.uniform(ObstacleGenerator.min_position.x, ObstacleGenerator.max_position.x)
                new_obstacles.append(Obstacle(obs.size, Obstacle.Position(new_x, obs.position.y, obs.position.z, obs.position.r)))
            elif action == 'adjust_y':
                new_y = random.uniform(ObstacleGenerator.min_position.y, ObstacleGenerator.max_position.y)
                new_obstacles.append(Obstacle(obs.size, Obstacle.Position(obs.position.x, new_y, obs.position.z, obs.position.r)))
            elif action == 'resize':
                new_size = Obstacle.Size(
                    l=random.uniform(ObstacleGenerator.min_size.l, ObstacleGenerator.max_size.l),
                    w=random.uniform(ObstacleGenerator.min_size.w, ObstacleGenerator.max_size.w),
                    h=random.uniform(ObstacleGenerator.min_size.h, ObstacleGenerator.max_size.h)
                )
                new_obstacles.append(Obstacle(new_size, obs.position))
            elif action == 'rotate':
                new_r = random.uniform(ObstacleGenerator.min_position.r, ObstacleGenerator.max_position.r)
                new_obstacles.append(Obstacle(obs.size, Obstacle.Position(obs.position.x, obs.position.y, obs.position.z, new_r)))
        return new_obstacles

    @staticmethod
    def check_overlap(obstacles: List[Obstacle]) -> bool:
        for i, obs1 in enumerate(obstacles):
            for obs2 in obstacles[i + 1:]:
                if ObstacleGenerator.is_overlapping(obs1, obs2):
                    return True
        return False

    @staticmethod
    def is_overlapping(obs1: Obstacle, obs2: Obstacle) -> bool:
        dx = abs(obs1.position.x - obs2.position.x)
        dy = abs(obs1.position.y - obs2.position.y)
        overlap_x = dx < (obs1.size.l / 2 + obs2.size.l / 2)
        overlap_y = dy < (obs1.size.w / 2 + obs2.size.w / 2)
        return overlap_x and overlap_y


class QTable:
    """Q-table for storing and updating state-action values."""

    def __init__(self, alpha: float, gamma: float) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.q_values: Dict[Tuple, Dict[str, float]] = {}
        self.action_counts: Dict[Tuple, Dict[str, int]] = {}

    def get_q_value(self, state: Tuple, action: str) -> float:
        return self.q_values.get(state, {}).get(action, 0)

    def update(self, state: Tuple, action: str, reward: float, next_state: Tuple) -> None:
        old_value = self.get_q_value(state, action)
        next_max = max(self.q_values.get(next_state, {}).values(), default=0)
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

        if state not in self.q_values:
            self.q_values[state] = {}
        self.q_values[state][action] = new_value

        if state not in self.action_counts:
            self.action_counts[state] = {}
        self.action_counts[state][action] = self.action_counts[state].get(action, 0) + 1


class UCBPolicy:
    """Upper Confidence Bound policy for selecting actions."""

    def __init__(self, c: float) -> None:
        self.c = c

    def select_action(self, q_table: QTable, state: Tuple, actions: List[str], total_attempts: int) -> str:
        q_values = q_table.q_values.get(state, {})
        action_counts = q_table.action_counts.get(state, {})

        ucb_values = {}
        for action in actions:
            if action_counts.get(action, 0) == 0:
                return action  # Select untried action for exploration
            q_value = q_values.get(action, 0)
            count = action_counts.get(action, 1)
            ucb_values[action] = q_value + self.c * math.sqrt(math.log(total_attempts) / count)

        return max(ucb_values, key=ucb_values.get)


class UCBGenerator:
    """Main generator for UCB-based test case generation."""

    def __init__(self, case_study_file: str, alpha=0.1, gamma=0.99, c=1 / math.sqrt(2)) -> None:
        self.case_study = DroneTest.from_yaml(case_study_file)
        self.q_table = QTable(alpha, gamma)
        self.ucb_policy = UCBPolicy(c)

    def calculate_reward(self, test: TestCase, num_obstacles: int) -> float:
        min_dist = min(test.get_distances())
        if min_dist < 0.25:
            return 5 + (3 - num_obstacles)
        elif 0.25 <= min_dist < 1:
            return 2 + (3 - num_obstacles)
        elif 1 <= min_dist < 1.5:
            return 1 + (3 - num_obstacles)
        return 0

    def generate(self, budget: int) -> List[TestCase]:
        test_cases = []
        actions = ['adjust_x', 'adjust_y', 'resize', 'rotate']
        total_attempts = 0

        for _ in range(budget):
            obstacles = [ObstacleGenerator.create_random_obstacle() for _ in range(random.choice([1, 2, 3]))]
            while ObstacleGenerator.check_overlap(obstacles):
                obstacles = [ObstacleGenerator.create_random_obstacle() for _ in range(random.choice([1, 2, 3]))]

            state = self.get_state(obstacles)
            action = self.ucb_policy.select_action(self.q_table, state, actions, total_attempts)

            obstacles = ObstacleGenerator.apply_action(action, obstacles)
            while ObstacleGenerator.check_overlap(obstacles):
                obstacles = ObstacleGenerator.apply_action(action, obstacles)

            test = TestCase(self.case_study, obstacles)

            try:
                test.execute()
                reward = self.calculate_reward(test, len(obstacles))
                self.q_table.update(state, action, reward, self.get_state(obstacles))

                test.plot()
                test_cases.append(test)
                total_attempts += 1
            except Exception as e:
                print("Exception during test execution, skipping the test")
                print(e)

        return test_cases

    def get_state(self, obstacles: List[Obstacle]) -> Tuple:
        """Converts a list of obstacles into a state representation."""
        return tuple((obs.position.x, obs.position.y, obs.size.l, obs.size.w, obs.size.h, obs.position.r) for obs in obstacles)


if __name__ == "__main__":
    generator = UCBGenerator("case_studies/mission1.yaml")
    generator.generate(3)
