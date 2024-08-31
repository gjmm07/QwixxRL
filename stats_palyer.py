import numpy as np
from environment import Environment, _GameEnvironment, BOARD
from scipy import stats
from _Player import _Player


class Agent(_Player):

    def __init__(self,
                 name: str):
        super().__init__(name)
        self.dice_values = stats.norm.pdf(np.arange(2, 13), 7, 2.4152429348524986)
        self.dice_values = (self.dice_values - min(self.dice_values)) / (max(self.dice_values) - min(self.dice_values))

    def do_main_move(self, white_dr: int, color_dr: np.ndarray):
        """
        The agent does its main move after the dice were thrown -- first check colored dice roll then white dice roll
        :param white_dr: sum of the white dice
        :param color_dr: sum of each white with each colored dice
        :return:
        """
        # print(white_dr)
        # print(color_dr)
        # CHECK COLORED DICE ROLLS
        col_move_taken = self._check_colored_dice_rolls(color_dr)
        if self.env._game_env.is_game_over:
            return
        # # CHECK WHITE DICE ROLL
        w_move_taken = self._check_white_dice_roll(white_dr)
        if not any([w_move_taken, col_move_taken]):
            print(self.name, "error taken")
            self.env.take_error()

    def downstream_move(self, white_dr: int) -> bool:
        return self._check_white_dice_roll(white_dr)

    def _check_white_dice_roll(self, white_dr: int, alpha: float = 0.5, threshold: float = 0.5) -> bool:
        """
        Based on the left border of the environment and how valuable a dice roll is (2 and 12 more valuable than 7)
        a move is taken based on the white dice roll
        :param white_dr: sum of the two white dices
        :return: move taken
        """
        dists = self.env.dists
        dists += 1
        norm_dist = (np.where(BOARD - white_dr == 0)[1] - dists) / 10
        values = self.dice_values[[white_dr - 2] * 4] * alpha + norm_dist * (1 - alpha)
        if np.all((values > threshold) | (norm_dist < 0)):
            return False
        valid_idx = np.where(values > 0)[0]
        idx = np.random.choice(valid_idx[values[valid_idx] == values[valid_idx].min()])
        self.env.take_move(idx, white_dr)
        print("white ", idx, white_dr)
        return True

    def _check_colored_dice_rolls(self, color_dr: np.ndarray, alpha: float = 0.5, threshold: float = 0.5) -> bool:
        """
        Based on the left border of the environment and how valuable a dice roll is (2 and 12 more valuable than 7)
        a move is taken based on the colored dice rolls
        :param color_dr: numpy array containing the sum of each white dice with each colored dice
        :return: move taken
        """
        dists = self.env.dists
        # dists[dists == -1] = -2
        dists += 1
        left_dist = np.where(np.stack((BOARD.T, BOARD.T), axis=2) - color_dr == 0)
        norm_dist = (left_dist[0] - dists[left_dist[1]]) / 10
        values = self.dice_values[color_dr[left_dist[1], left_dist[2]] - 2] * alpha + norm_dist * (1 - alpha)
        if np.all(values > threshold): # todo: or norm dist is smaller than one
            return False
        valid_idx = np.where(norm_dist >= 0)[0]
        idx = np.random.choice(valid_idx[values[valid_idx] == values[valid_idx].min()])
        self.env.take_move(left_dist[1][idx], BOARD[left_dist[1][idx], left_dist[0][idx]])
        print("color ", left_dist[1][idx], BOARD[left_dist[1][idx], left_dist[0][idx]])
        return True

    def print_total_score(self):
        print(self.name)
        print(self.env.sel_fields)
        print(self.env.error_count)
        print(self.env.compute_total_score())

    @property
    def is_real(self):
        return False

