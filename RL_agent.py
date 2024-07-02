from __future__ import annotations

import shutil
import typing

import numpy as np
import pickle
import os
from Player import Player
from environment import ExtraType
import random

C = 1.41


class Node:

    def __init__(self,
                 action: tuple[tuple[int, int], ...] | None,
                 parent=None,
                 id_: int = 0):
        self.parent: Node | None = parent
        self._id = id_
        self.action: tuple[tuple[int, int], ...] = action
        self._children: list[Node] = []
        self.n: int = 0
        self.q: float = 0.0

    @property
    def children(self):
        return self._children

    def set_children(self, children: list[Node]) -> None:
        self._children = children

    def add_child(self, child: Node):
        self._children.append(child)

    def get_best_child(self,
                       pos_actions: list[tuple[int, int] | ExtraType]) -> Node:
        idx = np.argmax([child.value for child in self._children if child.action in pos_actions])
        return self._children[idx]

    def get_non_expanded(self, actions: tuple, mask, extra_val) -> list[tuple]:
        child_actions = [child.action for child in self._children]
        possible_actions = [tuple(a) for a, m in zip(actions, mask) if m] + [extra_val]
        return [action for action in possible_actions if action not in child_actions]

    def save(self, path: os.PathLike or str):
        path = os.path.join(path, str(self._id) + ".pickle")
        save_data = {"n": self.n,
                     "q": self.q,
                     "id": self._id,
                     "action": self.action,
                     "children_id": [child._id for child in self._children]}
        if self.parent is not None:
            save_data["parent"] = self.parent._id
        pickle.dump(save_data, open(path, "wb"))

    @property
    def value(self):
        return self.q / self.n + C * np.sqrt(2 * np.log(self.parent.n) / self.n)


def get_random_action(possible_actions: list[list[tuple[int, int]]],
                      extra: ExtraType,
                      mask: list[bool] | None = None) -> list[tuple[int, int]]:
    if mask is None:
        mask = [True] * len(possible_actions)
    mask.append(True)
    possible_actions.append(extra.value)
    idx = np.random.choice(list(np.where(mask)[0]))
    return possible_actions[idx]


class RLAgent(Player):

    def __init__(self, name: str):
        super().__init__(name)
        self.base_node: Node = Node(None)
        self.parent_node: Node = self.base_node
        self.path: list[Node] = [self.parent_node]
        self._node_no: int = 1

    def add_node(self, actions: tuple[tuple[int, int]]):
        node = Node(actions, parent=self.parent_node, id_=self._node_no)
        self._node_no += 1
        self.path.append(node)
        self.parent_node.add_child(node)
        self.parent_node = node

    def do_main_move(self, white_dr: int, color_dr: np.ndarray) -> tuple[list[tuple[int, int]]] or None:
        possible_actions, mask = self.env.get_possible_actions(white_dr, color_dr)
        non_expanded_actions = self.parent_node.get_non_expanded(possible_actions, mask, ExtraType.error)
        if not non_expanded_actions:
            node = self.parent_node.get_best_child(
                [tuple(pos_act) for pos_act, is_allowed in zip(possible_actions, mask) if is_allowed] +
                [ExtraType.error])
            self.path.append(node)
            self.parent_node = node
            actions = node.action
        else:
            actions = non_expanded_actions[random.choice(range(len(non_expanded_actions)))]
            self.add_node(actions)
        if actions == ExtraType.error:
            self.take_action(ExtraType.error)
            return
        self.take_action(actions)
        return actions

    def downstream_move(self, white_dr: int) -> tuple[list[tuple[int, int]]] or None:
        pos_white_actions, white_mask = self.env.get_possible_white_actions(white_dr)
        non_expanded_actions = self.parent_node.get_non_expanded(pos_white_actions, white_mask, ExtraType.blank)
        if not non_expanded_actions:
            node = self.parent_node.get_best_child(
                [tuple(pos_act) for pos_act, is_allowed in zip(pos_white_actions, white_mask) if is_allowed] +
                [ExtraType.blank])
            self.path.append(node)
            self.parent_node = node
            action = node.action
        else:
            action = non_expanded_actions[random.choice(range(len(non_expanded_actions)))]
            self.add_node(action)
        if action == ExtraType.blank:
            return
        self.take_action(action)
        return action

    def _get_total_score(self, print_to_console: bool = True):
        total_score = self.env.compute_total_score()
        if print_to_console:
            print(f"{self.name}: \t {total_score}")
        return total_score

    def end_game_callback(self):
        return self._get_total_score(False)

    def reset_callback(self):
        self.parent_node: Node = self.base_node
        self.path: list[Node] = [self.parent_node]
        self.env.reset()

    def backpropagation(self, score: int):
        for node in self.path:
            node.n += 1
            node.q += score

    def _save_node(self, child: Node, path: os.PathLike or str):
        child.save(path)
        for c in child.children:
            self._save_node(c, path)

    def save_model(self):
        path = f"./models/{self.name}"
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=False)
        self._save_node(self.parent_node, path)
        print("DONE")

    def read_model(self):
        path = f"./models/{self.name}"
        nodes = {}
        for _, _, files in os.walk(path):
            for file in files:
                saved_data = pickle.load(open(os.path.join(path, file), "rb"))
                if len(saved_data["children_id"]) > 1:
                    print(len(saved_data.keys()))
                nodes[saved_data["id"]] = (Node(saved_data["action"], id_=saved_data["id"]), saved_data)
        for id_, (node, sd) in nodes.items():
            if id_ != 0:
                node.parent = nodes[sd["parent"]][0]
            node.set_children([nodes[child_id][0] for child_id in sd["children_id"]])
        base_node = nodes[0][0]
        print(base_node.children)

    @property
    def is_real(self):
        return False


if __name__ == "__main__":
    rla = RLAgent("Finn")
    rla.read_model()

