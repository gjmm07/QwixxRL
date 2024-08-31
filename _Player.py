import numpy as np

from environment import Environment, ExtraType


class _Player:

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

    def do_main_move(self, *args, **kwargs):
        raise NotImplementedError

    def downstream_move(self, dice_roll: np.ndarray):
        raise NotImplementedError

    def end_game_callback(self, *args, **kwargs) -> int:
        raise NotImplementedError

    def end_round_callback(self, *args, **kwargs):
        raise NotImplementedError

    def start_round_callback(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def is_game_over(self):
        return self.env.is_game_over
