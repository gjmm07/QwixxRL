from __future__ import annotations

import typing

import numpy as np
from dataclasses import dataclass, field
from dice_roll import throw_dice
from typing import Any, Generator, Literal, Union
from collections import deque
from environment import Environment, BOARD, _GameEnvironment
from Player import Player
from environment import ExtraType


class Node:

    def __init__(self, parent=None):
        self.parent: Node | None = parent
        self._children: dict[tuple[tuple[int, int], ...], Node] = dict()
        self.n: int = 0
        self.q: float = 0.0

    @property
    def children(self) -> dict[tuple[tuple[int, int], ...], Node]:
        return self._children

    def get_node(self, action) -> Node:
        return self.children[action]

    def add_child(self, action: tuple, child: Node):
        self._children[action] = child

    @property
    def value(self):
        return self.q / self.n + 2 * np.sqrt(2 * np.log(self.parent.n) / self.n)


@dataclass
class MCTree:
    base_node: Node = Node()
    tree: set[Node] = field(init=False)

    def __post_init__(self):
        self.tree = {self.base_node}


def get_random_action(possible_actions: list[list[tuple[int, int]]],
                      mask: list[bool],
                      extra: ExtraType) -> list[tuple[int, int]]:
    mask.append(True)
    possible_actions.append(extra.value)
    idx = np.random.choice(list(np.where(mask)[0]))
    return possible_actions[idx]


TREE = MCTree()


class RLAgent(Player):

    def __init__(self, name: str):
        super().__init__(name)
        self.parent_node: Node = TREE.base_node
        self.path: list[Node] = [self.parent_node]

    def add_node(self, actions: list[tuple[int, int]]):
        actions = tuple(actions)
        if actions in list(self.parent_node.children):
            node = self.parent_node.get_node(actions)
        else:
            node = Node(parent=self.parent_node)
            TREE.tree.add(node)
        self.path.append(node)
        self.parent_node.add_child(actions, node)
        self.parent_node = node

    def do_main_move(self, white_dr: int, color_dr: np.ndarray) -> tuple[list[tuple[int, int]]] or None:
        possible_actions, mask = self.env.get_possible_actions(white_dr, color_dr)
        actions = get_random_action(possible_actions, mask, ExtraType.error)
        if actions == ExtraType.error.value:
            self.take_action(ExtraType.error)
            return
        self.take_action(actions)
        self.add_node(actions)
        return actions

    def downstream_move(self, white_dr: int) -> tuple[list[tuple[int, int]]] or None:
        pos_white_actions, white_mask = self.env.get_possible_white_actions(white_dr)
        action = get_random_action(pos_white_actions, white_mask, ExtraType.blank)
        if action == ExtraType.blank.value:
            return
        self.take_action(action)
        self.add_node(action)
        return action

    def _get_total_score(self, print_to_console: bool = True):
        total_score = self.env.compute_total_score()
        if print_to_console:
            print(self.name)
            print(self.env.sel_fields)
            print(self.env.error_count)
            print(total_score)
        return total_score

    def end_game_callback(self):
        print(self.path)
        return self._get_total_score()

    def reset_callback(self):
        self.parent_node: Node = TREE.base_node
        self.path: list[Node] = [self.parent_node]
        self.env.reset()

    def backpropagation(self, score: int):
        print("len tree: ", len(TREE.tree))

    @property
    def is_real(self):
        return False

if __name__ == "__main__":
    TREE.tree.add(Node((1, 2)))
    print(len(TREE.tree))
