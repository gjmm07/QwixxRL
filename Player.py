import numpy as np

from environment import Environment, ExtraType


class Player:

    def __init__(self, name: str):
        self.name = name
        self.env = Environment()

    def take_action(self,
                    moves: list[tuple[int, int]] | ExtraType,
                    first_player: bool = False,
                    white_dr: int = 0,
                    color_dr: np.ndarray = None):
        if isinstance(moves, ExtraType):
            assert not first_player and moves == ExtraType.error, "No error if white move"
            if moves == ExtraType.error:
                self.env.take_error()
            return
        self.env.take_move_idx(*list(zip(*moves)))
