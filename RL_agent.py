from __future__ import annotations
import numpy as np
from dice_roll import throw_dice
from typing import Any, Generator
from collections import deque
from environment import Environment, BOARD, GameEnvironment


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

    def get_action(self, possible_actions: list[list[tuple]], mask: list[bool]) -> list[tuple]:
        idx = np.random.choice(list(np.where(mask)[0]) + [-1])
        if idx == -1:
            # self.add_node([idx, ])
            return idx
        # self.add_node(possible_actions[idx])
        return possible_actions[idx]

    def restart(self):
        self.parent_node = self.root


class RLAgent:

    mcts = MonteCarloTree()

    def __init__(self, name, environment: Environment):
        self.env = environment
        self.name = name
        self.path: list[Node] = []

    def do_main_move(self, white_dr: int, color_dr: np.ndarray):
        # print("do main move")
        possible_actions, mask = self.env.get_possible_actions(white_dr, color_dr)
        # print(list(zip(possible_actions, mask)))
        actions = self.mcts.get_action(possible_actions, mask)
        if actions == -1:
            # print("take error")
            self.env.take_error()
            return None
        for action in actions:
            self.env.take_move(action[0], BOARD[action])
        return actions

    def downstream_move(self, white_dr: int):
        # print("downstream move")
        # todo: What about downstream move?
        pos_white_actions, white_mask = self.env.get_possible_white_actions(white_dr)
        action = self.mcts.get_action(list(zip(*pos_white_actions)), white_mask)
        if action == -1:
            return None
        self.env.take_move(action[0], BOARD[action])
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


def next_player(players: deque[RLAgent]) -> Generator[None | RLAgent, None, None]:
    # todo: ensure players start opposite seq when a new game begins
    yield None
    while True:
        for player in players:
            yield player
        players.rotate(-1)
        yield None


def main():
    for _ in range(100):
        print("_______________")
        white_dr: int = 0
        game_environment = GameEnvironment()
        agents: deque[RLAgent] = deque([RLAgent("RL1", Environment(game_environment)),
                                        RLAgent("RL2", Environment(game_environment))])
        gen = next_player(agents)
        while not game_environment.is_game_over:
            agent = next(gen)
            if agent is None:
                agent = next(gen)
                _, white_dr, color_dr = throw_dice()
                # print(white_dr)
                # print(color_dr)
                agent.do_main_move(white_dr, color_dr)
                if game_environment.is_game_over:
                    break
            else:
                agent.downstream_move(white_dr)
        final_scores = []
        for agent in agents:
            final_scores.append(agent.print_total_score())
        # for agent, score in zip(agents, (np.array(final_scores) == max(final_scores)).astype(int)):
        #     agent.restart(score)
        # game_environment.reset()


if __name__ == "__main__":
    main()
    # agent.env.take_move(1, 2)
    # agent.env.take_move(1, 3)
    # agent.env.take_move(1, 4)
    # print(agent.env.sel_fields)
    # w_dr, c_dr = _throw_dice()
    # w_dr, c_dr = 8, np.array([[8, 12],
    #                           [8, 12],
    #                           [7, 6],
    #                           [10, 9]])
    #
    # agent._get_possible_actions(w_dr, c_dr)
