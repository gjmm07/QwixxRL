from __future__ import annotations
import numpy as np
from dice_roll import throw_dice
from typing import Any, Generator, Literal, Union
from collections import deque
from environment import Environment, BOARD, _GameEnvironment
from Player import Player
from environment import ExtraType


class Node:

    def __init__(self, parent=None):
        self.parent: Node | None = parent
        self.children: dict[tuple, Node] = {}
        self.n: int = 0
        self.q: float = 0.0

    def add_child(self, action: tuple, child: Node):
        self.children[action] = child

    @property
    def value(self):
        return self.q / self.n + 2 * np.sqrt(2 * np.log(self.parent.n) / self.n)


class MonteCarloTree:

    def __init__(self):
        self.parent_node = Node()
        self.root = self.parent_node
        self._tree = [self.parent_node]

    def add_node(self, action: list[tuple] | None):
        action = tuple(action)
        if action in self.parent_node.children.keys():
            self.parent_node = self.parent_node.children[action]
            return
        node = Node(self.parent_node)
        self._tree.append(node)
        self.parent_node.add_child(action, node)
        self.parent_node = node

    def get_action(self,
                   possible_actions: list[list[tuple[int, int]]],
                   mask: list[bool],
                   extra: ExtraType) -> list[tuple[int, int]]:
        idx = np.random.choice(list(np.where(mask)[0]) + [extra.value])
        if idx == extra:
            return idx
        return possible_actions[idx]

    def restart(self):
        self.parent_node = self.root


class RLAgent(Player):

    mcts = MonteCarloTree()

    def __init__(self, name):
        super().__init__(name)
        self.path: list[Node] = []

    def do_main_move(self, white_dr: int, color_dr: np.ndarray):
        # print("do main move")
        possible_actions, mask = self.env.get_possible_actions(white_dr, color_dr)
        # print(list(zip(possible_actions, mask)))
        actions = self.mcts.get_action(possible_actions, mask, ExtraType.error)
        if actions == ExtraType.error.value:
            self.take_action(ExtraType.error)
            return
        self.take_action(actions)
        return actions

    def downstream_move(self, white_dr: int):
        pos_white_actions, white_mask = self.env.get_possible_white_actions(white_dr)
        action = self.mcts.get_action(pos_white_actions, white_mask, ExtraType.blank)
        if action == ExtraType.blank.value:
            return
        self.take_action(ExtraType.blank)
        self.take_action(action)
        return action

    def print_total_score(self):
        print(self.name)
        print(self.env.sel_fields)
        print(self.env.error_count)
        total_score = self.env.compute_total_score()
        print(total_score)
        return total_score

    def restart(self, score: int):
        print([node.children.keys() for node in self.path])
        print(score)
        self.path = []
        self.mcts.restart()
        self.env.reset()

    @property
    def is_real(self):
        return False
