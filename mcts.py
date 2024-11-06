"""
    Copied From TUMB at the SBFT 2024 Tool Competition - CPS-UAV Test Case Generation Track
    https://github.com/MayDGT/UAV-Testing-Competition
"""
import math
import random
import datetime
import logging
from typing import List
from scenarioState import ScenarioState
from testcase import TestCase
import sys
import os

class Node:
    def __init__(self, state: ScenarioState, parent):
        self.state = state
        self.parent = parent
        self.visits = 0
        self.reward = 0.0
        self.children = []

    def __str__(self):
        return f"state: \n {str(self.state)}, visits: {self.visits}, reward: {self.reward}"

class MCTS:
    def __init__(self, case_study_file: str)-> None:
        self.initial_state = ScenarioState(case_study_file)
        self.root = Node(self.initial_state, None)

        # hyperparameters for UCB1 and progressive widening
        self.exploration_rate = 1 / math.sqrt(2)
        self.C = 0.5
        self.alpha = 0.5
        # self.C_list = [0.4, 0.5, 0.6, 0.7]
        self.C_list = [0.4, 0.5, 0.6]

        self.test_cases = []

    def select(self, node: Node):
        while not node.state.is_terminal():
            layer = len(node.state.scenario)
            # progressive widening
            if len(node.children) <= self.C_list[layer] * (node.visits ** self.alpha):
                return self.expand(node)
            else:
                node = self.best_child(node)

        return node

    @staticmethod
    def expand(node: Node):
        tried_children_state = [c.state for c in node.children]
        new_state = node.state.next_state()
        while new_state in tried_children_state and not new_state.is_terminal():
            new_state = node.state.next_state()
        if len(node.state.scenario) == len(new_state.scenario):
            return None
        else:
            new_node = Node(new_state, node)
            node.children.append(new_node)
            return new_node

    def simulate(self, state):
        return state.get_reward()

    @staticmethod
    def back_propogate(node, reward):
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent

    def search(self):
        node = self.select(self.root)
        if node is not None:
            reward, min_distance, test_case = self.simulate(node.state)
            # delete the node if it is invalid
            if reward == 0.0 and len(node.state.scenario) != 0:
                node.parent.children.remove(node)
                node.parent = None

            if 0 <= abs(min_distance) <= 1.5:
                self.test_cases.append(test_case)

            self.back_propogate(node, reward)

    def generate(self, budget: int) -> List[TestCase]:
        reward, distance, test_case = self.simulate(self.root.state)
        self.back_propogate(self.root, reward)
        for i in range(budget):
            self.search()
        return self.test_cases

    def best_child(self, node):
        # UCB1
        return max(node.children, key=lambda child: child.reward / child.visits +
                                             self.exploration_rate * math.sqrt(2 * math.log(node.visits) / child.visits))


if __name__ == '__main__':
    pass



