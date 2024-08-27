from Player import Player
import numpy as np
from environment import ExtraType


class DQLAgent(Player):

    def __init__(self, name: str):
        super().__init__(name)

    def do_main_move(self, white_dr: int, color_dr: np.ndarray):
        # (4, 2) single color actions, 4 white actions, combo moves and error = 45 possible moves
        moves, allowed = self.env.get_possible_actions(white_dr, color_dr)
        allowed += [True]
        moves += [ExtraType.error]
        action = moves[np.random.choice(np.where(allowed)[0])]
        super().take_action(action)
        print(action)
        return action

    def downstream_move(self, white_dr: int):
        # 4 white actions and blank moves = 5 possible moves
        moves, allowed = self.env.get_possible_white_actions(white_dr)
        allowed += [True]
        moves += [ExtraType.blank]
        action = moves[np.random.choice(np.where(allowed)[0])]
        super().take_action(action)
        return action

    def end_round_callback(self):
        print("end round")

    def end_game_callback(self):
        self.env.reset()
        print("end game")

    @property
    def is_real(self):
        return False
