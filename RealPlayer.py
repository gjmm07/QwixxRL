import numpy as np
from environment import ExtraType
from Player import Player


class RealPlayer(Player):

    CLOSED_ROWS = []

    def __init__(self,
                 name: str):
        super().__init__(name)

    def _get_total_score(self):
        print(self.name)
        total_score = self.env.compute_total_score()
        print(total_score)
        return total_score

    def take_action(self,
                    moves: list[tuple[int, int]] | ExtraType,
                    first_player: bool = False,
                    white_dr: int = 0,
                    color_dr: np.ndarray = None):
        # todo: check if move is allowed
        print(moves)
        if ExtraType.error in moves:
            super().take_action(ExtraType.error)
            return
        elif not first_player and (ExtraType.blank in moves or not moves):
            return
        if first_player:
            poss_moves, mask = self.env.get_possible_actions(white_dr, color_dr)
        else:
            poss_moves, mask = self.env.get_possible_white_actions(white_dr)
        assert set(moves) in (set(pm) for pm, contain in zip(poss_moves, mask) if contain), "wrong selection"
        super().take_action(moves)

    def end_game_callback(self):
        return self._get_total_score()

    def reset_callback(self):
        self.env.reset()

    @property
    def is_real(self):
        return True

