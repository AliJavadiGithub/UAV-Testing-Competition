import random
import math
from typing import List
from aerialist.px4.drone_test import DroneTest
from aerialist.px4.obstacle import Obstacle
from testcase import TestCase

class UCBGenerator:
    min_size = Obstacle.Size(2, 2, 15)
    max_size = Obstacle.Size(20, 20, 25)
    min_position = Obstacle.Position(-40, 10, 0, 0)
    max_position = Obstacle.Position(30, 40, 0, 90)

    def __init__(self, case_study_file: str, alpha=0.1, gamma=0.99, c=1 / math.sqrt(2)) -> None:
        self.case_study = DroneTest.from_yaml(case_study_file)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.c = c  # Exploration parameter for UCB
        self.q_table = {}  # Q-table for storing state-action values
        self.action_counts = {}  # Count of each action taken in each state

    def get_state(self, obstacles):
        return tuple((obs.position.x, obs.position.y, obs.size.l, obs.size.w, obs.size.h, obs.position.r) for obs in obstacles)

    def choose_action(self, state, actions, total_attempts):
        q_values = self.q_table.get(state, {})
        action_counts = self.action_counts.get(state, {})
        
        ucb_values = {}
        for action in actions:
            if action_counts.get(action, 0) == 0:
                return action  # Select untried action for exploration
            q_value = q_values.get(action, 0)
            count = action_counts.get(action, 1)
            ucb_values[action] = q_value + self.c * math.sqrt(math.log(total_attempts) / count)
        
        return max(ucb_values, key=ucb_values.get)

    def calculate_reward(self, test, num_obstacles):
        distances = test.get_distances()
        min_dist = min(distances)

        if min_dist < 0.25:
            return 5 + (3 - num_obstacles)
        elif 0.25 <= min_dist < 1:
            return 2 + (3 - num_obstacles)
        elif 1 <= min_dist < 1.5:
            return 1 + (3 - num_obstacles)
        else:
            return 0

    def generate(self, budget: int) -> List[TestCase]:
        test_cases = []
        actions = ['adjust_x', 'adjust_y', 'resize', 'rotate']
        total_attempts = 0
        
        for _ in range(budget):
            obstacles = [self.random_obstacle() for _ in range(random.choice([1, 2, 3]))]
            while self.check_overlap(obstacles):
                obstacles = [self.random_obstacle() for _ in range(random.choice([1, 2, 3]))]
                
            state = self.get_state(obstacles)
            action = self.choose_action(state, actions, total_attempts)
            
            obstacles = self.apply_action(action, obstacles)
            while self.check_overlap(obstacles):
                obstacles = self.apply_action(action, obstacles)
                
            test = TestCase(self.case_study, obstacles)
            
            try:
                test.execute()
                num_obstacles = len(obstacles)
                reward = self.calculate_reward(test, num_obstacles)
                self.update_q_table(state, action, reward, obstacles)
                
                test.plot()
                test_cases.append(test)
                total_attempts += 1
            except Exception as e:
                print("Exception during test execution, skipping the test")
                print(e)
        
        return test_cases

    def apply_action(self, action, obstacles):
        for obs in obstacles:
            if action == 'adjust_x':
                new_x = random.uniform(self.min_position.x, self.max_position.x)
                obs.position = Obstacle.Position(new_x, obs.position.y, obs.position.z, obs.position.r)
            elif action == 'adjust_y':
                new_y = random.uniform(self.min_position.y, self.max_position.y)
                obs.position = Obstacle.Position(obs.position.x, new_y, obs.position.z, obs.position.r)
            elif action == 'resize':
                new_l = random.uniform(self.min_size.l, self.max_size.l)
                new_w = random.uniform(self.min_size.w, self.max_size.w)
                new_h = random.uniform(self.min_size.h, self.max_size.h)
                obs.size = Obstacle.Size(new_l, new_w, new_h)
            elif action == 'rotate':
                new_r = random.uniform(self.min_position.r, self.max_position.r)
                obs.position = Obstacle.Position(obs.position.x, obs.position.y, obs.position.z, new_r)
        return obstacles

    def random_obstacle(self):
        size = Obstacle.Size(
            l=random.uniform(self.min_size.l, self.max_size.l),
            w=random.uniform(self.min_size.w, self.max_size.w),
            h=random.uniform(self.min_size.h, self.max_size.h),
        )
        position = Obstacle.Position(
            x=random.uniform(self.min_position.x, self.max_position.x),
            y=random.uniform(self.min_position.y, self.max_position.y),
            z=0,
            r=random.uniform(self.min_position.r, self.max_position.r),
        )
        return Obstacle(size, position)

    def check_overlap(self, obstacles):
        for i, obs1 in enumerate(obstacles):
            for obs2 in obstacles[i + 1:]:
                if self.is_overlapping(obs1, obs2):
                    return True
        return False

    def is_overlapping(self, obs1, obs2):
        dx = abs(obs1.position.x - obs2.position.x)
        dy = abs(obs1.position.y - obs2.position.y)
        
        overlap_x = dx < (obs1.size.l / 2 + obs2.size.l / 2)
        overlap_y = dy < (obs1.size.w / 2 + obs2.size.w / 2)
        
        return overlap_x and overlap_y

    def update_q_table(self, state, action, reward, obstacles):
        next_state = self.get_state(obstacles)
        old_q_value = self.q_table.get(state, {}).get(action, 0)
        next_max = max(self.q_table.get(next_state, {}).values(), default=0)
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * next_max - old_q_value)
        
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = new_q_value
        
        if state not in self.action_counts:
            self.action_counts[state] = {}
        self.action_counts[state][action] = self.action_counts[state].get(action, 0) + 1

if __name__ == "__main__":
    generator = UCBGenerator("case_studies/mission1.yaml")
    generator.generate(3)
