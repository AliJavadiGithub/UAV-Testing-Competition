import random
from typing import List
from aerialist.px4.drone_test import DroneTest
from aerialist.px4.obstacle import Obstacle
from testcase import TestCase

class QLearningTestGenerator(object):
    min_size = Obstacle.Size(2, 2, 15)
    max_size = Obstacle.Size(20, 20, 25)
    min_position = Obstacle.Position(-40, 10, 0, 0)
    max_position = Obstacle.Position(30, 40, 0, 90)

    def __init__(self, case_study_file: str, alpha=0.1, gamma=0.9, epsilon=0.2) -> None:
        self.case_study = DroneTest.from_yaml(case_study_file)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}  # Q-table for storing state-action values

    def get_state(self, obstacles):
        # Convert obstacle configuration to a tuple representing state
        return tuple((obs.position.x, obs.position.y, obs.size.l, obs.size.w, obs.size.h, obs.position.r) for obs in obstacles)

    def choose_action(self, state, actions):
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(actions)
        q_values = self.q_table.get(state, {})
        return max(q_values, key=q_values.get, default=random.choice(actions))

    def generate(self, budget: int) -> List[TestCase]:
        test_cases = []
        actions = ['adjust_x', 'adjust_y', 'resize', 'rotate']
        for _ in range(budget):
            obstacles = [self.random_obstacle() for _ in range(random.choice([1, 2, 3]))]
            state = self.get_state(obstacles)
            action = self.choose_action(state, actions)
            
            # Execute action on obstacles
            obstacles = self.apply_action(action, obstacles)
            test = TestCase(self.case_study, obstacles)
            
            try:
                test.execute()
                distances = test.get_distances()
                min_distance = min(distances)
                
                # Reward failed tests with fewer obstacles  
                reward = 100 if 0 <= abs(min_distance) <= 1.5 else -min_distance  # Reward for failure or deviation 
                self.update_q_table(state, action, reward, obstacles)
                
                test.plot()
                test_cases.append(test)
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
            z=0,  # obstacles are always on the ground
            r=random.uniform(self.min_position.r, self.max_position.r),
        )
        return Obstacle(size, position)

    def update_q_table(self, state, action, reward, obstacles):
        # Q-learning update rule
        next_state = self.get_state(obstacles)
        old_q_value = self.q_table.get(state, {}).get(action, 0)
        next_max = max(self.q_table.get(next_state, {}).values(), default=0)
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * next_max - old_q_value)
        
        # Update Q-table
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = new_q_value

if __name__ == "__main__":
    generator = QLearningTestGenerator("case_studies/mission1.yaml")
    generator.generate(3)