import numpy as np
from environment import ExtraType
from _Player import _Player


class RealPlayer(_Player):

    CLOSED_ROWS = []

    def __init__(self,
                 name: str):
        super().__init__(name)

    def _get_total_score(self):
        total_score = self.env.compute_total_score()
        return total_score

    def take_action(self,
                    moves: list[tuple[int, int]] | ExtraType):
        super().take_action(moves)

    def end_game_callback(self, *args, **kwargs) -> int:
        return self._get_total_score()

    def end_round_callback(self, *args, **kwargs):
        return

    def start_round_callback(self, *args, **kwargs):
        return

    def reset_callback(self):
        self.env.reset()

    @property
    def is_real(self):
        return True

