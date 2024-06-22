import numpy as np
from environment import ExtraType
from Player import Player


class RealPlayer(Player):

    CLOSED_ROWS = []

    def __init__(self,
                 name: str):
        super().__init__(name)

    def print_total_score(self):
        print(self.name)
        print(self.env.compute_total_score())

    def take_action(self,
                    moves: list[tuple[int, int]] | ExtraType,
                    first_player: bool = False,
                    white_dr: int = 0,
                    color_dr: np.ndarray = None):
        print(moves)
        if ExtraType.error in moves:
            self.env.take_error()
            return
        elif not first_player and (ExtraType.blank in moves or not moves):
            return
        if first_player:
            poss_moves, mask = self.env.get_possible_actions(white_dr, color_dr)
        else:
            poss_moves, mask = self.env.get_possible_white_actions(white_dr)
        assert moves in (pm for pm, contain in zip(poss_moves, mask) if contain), "wrong selection"
        super().take_action(moves)

    @property
    def is_real(self):
        return True

