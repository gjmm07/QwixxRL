import numpy as np
from environment import Environment
from scipy import stats


class Agent:

    ALPHA = 0.01  # to balance between left distance and dice value todo: improve?

    def __init__(self,
                 name: str,
                 env: Environment):
        self.name = name
        self.env = env
        self.dice_values = stats.norm.pdf(np.arange(2, 13), 7, 2.4152429348524986) + self.ALPHA

    def do_main_move(self, white_dr: int, color_dr: np.ndarray):
        """
        The agent does its main move after the dice were thrown -- first check colored dice roll then white dice roll
        :param white_dr: sum of the white dice
        :param color_dr: sum of each white with each colored dice
        :return:
        """
        print(white_dr)
        print(color_dr)
        # CHECK COLORED DICE ROLLS
        col_move_taken = self._check_colored_dice_rolls(color_dr)
        if self.is_game_over:
            return
        # CHECK WHITE DICE ROLL
        w_move_taken = self._check_white_dice_roll(white_dr)
        if not any([w_move_taken, col_move_taken]):
            print(self.name, "error taken")
            self.env.take_error()

    def downstream_move(self, white_dr: int) -> bool:
        return self._check_white_dice_roll(white_dr)

    def _check_white_dice_roll(self, white_dr: int) -> bool:
        """
        Based on the left border of the environment and how valuable a dice roll is (2 and 12 more valuable than 7)
        a move is taken based on the white dice roll
        :param white_dr: sum of the two white dices
        :return: move taken
        """
        cur_env = self.env.return_current_board()
        possible_moves = np.where((cur_env - white_dr) == 0)
        ind = np.indices(cur_env.shape)[1]
        mask = np.zeros_like(cur_env)
        mask[possible_moves[0], :] = ind[possible_moves[0], :] <= possible_moves[1][:, np.newaxis]
        left_dist = np.logical_and(mask, cur_env != -1).sum(axis=1)
        left_dist = left_dist[left_dist > 0]
        values = left_dist * self.dice_values[white_dr - 2]
        if any(values <= 0.3):
            idx = np.random.choice(np.where(values == values.min())[0])
            self.env.take_move(possible_moves[0][idx], white_dr)
            print(self.name, possible_moves[0][idx], white_dr, "white dice roll")
            return True
        return False

    def _check_colored_dice_rolls(self, color_dr: np.ndarray) -> bool:
        """
        Based on the left border of the environment and how valuable a dice roll is (2 and 12 more valuable than 7)
        a move is taken based on the colored dice rolls
        :param color_dr: numpy array containing the sum of each white dice with each colored dice
        :return: move taken
        """
        cur_env = self.env.return_current_board()
        dists = []
        for i in range(color_dr.shape[1]):
            poss_moves = np.where(cur_env - np.take(color_dr, [i], 1) == 0)
            mask = np.zeros_like(cur_env)
            mask[poss_moves[0], :] = np.indices(cur_env.shape)[1, poss_moves[0], :] <= poss_moves[1][:, np.newaxis]
            dists.append(np.logical_and(mask, cur_env != -1).sum(axis=1))
        dists = np.stack(dists, axis=1)
        valid_idx = np.where(dists > 0)
        print("________________")
        values = dists[valid_idx] * self.dice_values[color_dr[valid_idx] - 2]
        if np.any(values <= 0.3):
            idx = np.random.choice(np.where(values == values.min())[0])
            self.env.take_move(valid_idx[0][idx], color_dr[valid_idx[0][idx], valid_idx[1][idx]])
            print(self.name, valid_idx[0][idx], color_dr[valid_idx[0][idx], valid_idx[1][idx]], "color dice roll")
            return True
        return False

    def print_total_score(self):
        print(self.name)
        print(self.env.sel_fields)
        print(self.env.error_count)
        print(self.env.compute_total_score())

    @property
    def is_game_over(self) -> bool:
        return self.env.game_over


class RealPlayer:

    CLOSED_ROWS = []

    def __init__(self,
                 name: str,
                 env: Environment):
        self.name = name
        self.error_counter: int = 0
        self.game_over: bool = False
        self.env = env

    def close_row(self, idx: int):
        self.CLOSED_ROWS.append(idx)
        if len(self.CLOSED_ROWS) >= 2:
            self.game_over = True

    def take_error(self):
        self.error_counter += 1
        if self.error_counter >= 4:
            self.game_over = True

    def print_total_score(self):
        print(self.name)
        print(self.env.compute_total_score())

    @property
    def is_game_over(self) -> bool:
        return self.game_over


if __name__ == "__main__":
    agent = Agent("COM")
    agent.env.take_move(0, 3)
    agent.env.take_move(3, 6)
    agent.do_main_move(7, np.array([[3, 4], [5, 5], [12, 8], [12, 5]]))

