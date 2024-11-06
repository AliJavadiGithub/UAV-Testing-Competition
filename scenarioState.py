"""
    Copied From TUMB at the SBFT 2024 Tool Competition - CPS-UAV Test Case Generation Track
    https://github.com/MayDGT/UAV-Testing-Competition
"""
import os
import random
import logging
from copy import deepcopy
from typing import List
from aerialist.px4.drone_test import DroneTest
from aerialist.px4.obstacle import Obstacle
from testcase import TestCase
from utils import random_nonintersecting_rectangle, get_boundary_distance, random_rectangle, plot_rectangle

logger = logging.getLogger(__name__)


class ScenarioState:
    min_size = Obstacle.Size(2, 2, 15)
    max_size = Obstacle.Size(20, 20, 25)

    # fixed area: -40 < x < 30, 10 < y < 40
    min_position = Obstacle.Position(-40, 10, 0, 0)
    max_position = Obstacle.Position(30, 40, 0, 90)

    def __init__(self, mission_yaml=None, scenario: List[Obstacle] = []):
        self.scenario = scenario
        self.mission_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), mission_yaml)

        # drone's trajectory: [(x0,y0), (x1,y1), ...]
        self.trajectory = []

        self.min_reward = 0.0
        self.max_distance = 5.0
        self.max_obstacles = 3.0
        # self.max_obstacles = 4.0

    def next_state(self):
        """Generate a new obstacle on the path of the drone"""
        new_obstacle = self.generate()
        new_state = deepcopy(self)
        if new_obstacle is not None:
            new_state.scenario.append(new_obstacle)
        return new_state

    def get_reward(self):
        """Simulate the scenario and calculate the reward"""
        test = TestCase(DroneTest.from_yaml(self.mission_yaml), self.scenario)
        try:
            self.trajectory = [(position.x, position.y) for position in test.execute().positions]
        except Exception as e:
            return self.min_reward, self.max_distance, test

        if len(self.scenario) == 0:
            return self.min_reward, self.max_distance, test

        min_distance = min(test.get_distances())
        reward = -1.0 * min_distance
        test.plot()
        return reward, min_distance, test

    def is_terminal(self):
        # if len(self.scenario) == 4:
        if len(self.scenario) == 3:
            return True
        return False

    def generate(self):
        """Randomly choose a point on the drone's trajectory, as the center point of the new rectangle"""
        candidate_positions = []
        for position in self.trajectory:
            if len(self.scenario) == 0 and \
                    position[1] > self.min_position.y + 1/6 * (self.max_position.y - self.min_position.y):
                break
            if self.min_position.x < position[0] < self.max_position.x and \
                    self.min_position.y < position[1] < self.max_position.y:
                candidate_positions.append(position)

        candidate_position = random.choice(candidate_positions)
        if len(self.scenario) == 0:
            radius = get_boundary_distance(
                            candidate_position[0],
                            candidate_position[1],
                            self.max_position.y,
                            self.min_position.y,
                            self.min_position.x,
                            self.max_position.x
                        )
            x, y, l, w, r = random_rectangle(candidate_position[0], candidate_position[1], radius)
        else:
            x, y, l, w, r = random_nonintersecting_rectangle(
                                candidate_position[0],
                                candidate_position[1],
                                self.max_position.y,
                                self.min_position.y,
                                self.min_position.x,
                                self.max_position.x,
                                [(ob.position.x, ob.position.y, ob.size.l, ob.size.w, ob.position.r) for ob in self.scenario]
                            )
        position = Obstacle.Position(x, y, 0, r)
        size = Obstacle.Size(l, w, self.max_size.h)
        return Obstacle(size, position)

    def __eq__(self, other):
        list1 = [str(obstacle.to_dict()) for obstacle in self.scenario]
        list2 = [str(obstacle.to_dict()) for obstacle in other.scenario]
        return set(list1) == set(list2)

    def __str__(self):
        s = ""
        for ob in self.scenario:
            s += str((ob.position.x, ob.position.y, ob.size.l, ob.size.w, ob.position.r)) + '\n'
        return s


def replay(obstacles):
    state = ScenarioState()
    for obst in obstacles:
        position = Obstacle.Position(obst[0], obst[1], 0, obst[4])
        size = Obstacle.Size(obst[2], obst[3], 25)
        state.scenario.append(Obstacle(size, position))
    state.get_reward()


if __name__ == '__main__':
    pass

