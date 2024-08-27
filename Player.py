import numpy as np

from environment import Environment, ExtraType


class Player:

    def __init__(self, name: str):
        self.name = name
        self.env = Environment()

    def take_action(self,
                    moves: list[tuple[int, int]] | ExtraType):
        if moves == ExtraType.error:
            self.env.take_error()
            return
        if moves == ExtraType.blank:
            return
        self.env.take_move_idx(*list(zip(*moves)))

    @property
    def is_game_over(self):
        return self.env.is_game_over
