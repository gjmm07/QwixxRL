import numpy as np
from operator import itemgetter
from warnings import warn, showwarning
from dice_roll import throw_dice


BOARD: np.ndarray = np.array([np.arange(2, 13),
                              np.arange(2, 13),
                              np.arange(2, 13)[::-1],
                              np.arange(2, 13)[::-1]])


class GameEnvironment:

    def __init__(self):
        self.CLOSED_ROWS: list[int] = []
        self.PLAYER_ENDED_GAME: bool = False

    def reset(self):
        self.CLOSED_ROWS = []
        self.PLAYER_ENDED_GAME = False

    @property
    def is_game_over(self) -> bool:
        return self.PLAYER_ENDED_GAME


def _get_mask(dist, possible_moves, n_slc_row, closed_rows):
    # only allow actions right of possible action and only close row when enough selected
    mask = (dist < possible_moves[1]) & ~((n_slc_row < 4) & (possible_moves[1] == 10))
    if closed_rows:
        # disallow moves when row is closed
        mask = mask & np.any(possible_moves[0][:, np.newaxis] != np.array([closed_rows]), axis=1).reshape(-1)
    return mask


def _allowed_white_actions(n_slc_row: np.ndarray,
                           dist: np.ndarray,
                           white_dr: int,
                           closed_rows: list[int]) -> tuple[tuple, np.ndarray]:
    """
    Based on the white dice roll all possible actions are selected as well as their mask which ones are possible
    :param n_slc_row:
    :param dist:
    :param white_dr:
    :param closed_rows:
    :return:
    """
    possible_moves = np.where(BOARD == white_dr)
    mask = _get_mask(dist, possible_moves, n_slc_row, closed_rows)
    return possible_moves, mask


def _allowed_color_actions(n_slc_row: np.ndarray,
                           dist: np.array,
                           color_dr: np.ndarray,
                           closed_rows: list[int]) -> tuple[tuple, np.ndarray]:
    """
    Based on the color dice roll all possible actions are selected as well as their mask which ones are possible
    :param n_slc_row:
    :param dist:
    :param color_dr:
    :param closed_rows:
    :return:
    """
    possible_moves = (np.zeros((8,), dtype=int), np.zeros((8,), dtype=int))
    mask = np.zeros((8,), dtype=bool)
    for i in range(2):
        n_poss_moves = np.where(BOARD == color_dr[:, [i]])
        possible_moves[0][i * 4: i * 4 + 4], possible_moves[1][i * 4: i * 4 + 4] = n_poss_moves
        n_mask = _get_mask(dist, n_poss_moves, n_slc_row, closed_rows)
        mask[i * 4: i * 4 + 4] = n_mask
    return possible_moves, mask


def _allowed_combi_actions(pos_white_actions,
                           white_mask: np.ndarray,
                           pos_color_actions,
                           color_mask: np.ndarray,
                           error_count: int,
                           sel_field_row: np.ndarray):
    """
    Gets all possible combinations (white and color dices) and an array which masks only possible actions
    :param pos_white_actions:
    :param white_mask:
    :param pos_color_actions:
    :param color_mask:
    :param error_count:
    :param sel_field_row:
    :return:
    """
    pos_white_actions = np.array(pos_white_actions).T
    pos_color_actions = np.array(pos_color_actions).T
    pos_combi_actions = np.stack((np.tile(pos_white_actions, (len(pos_color_actions), 1)),
                                  np.repeat(pos_color_actions, len(pos_white_actions), axis=0)),
                                 axis=1)
    # take over from white and color possible moves -- special case (not) possible
    combi_mask = np.tile(white_mask, len(pos_color_actions)) & np.repeat(color_mask, len(pos_white_actions), axis=0)
    # Don't allow moves in same row if white is larger
    white_smaller = ((pos_combi_actions[:, 0, 0] != pos_combi_actions[:, 1, 0]) |
                     (pos_combi_actions[:, 0, 1] < pos_combi_actions[:, 1, 1]))
    combi_mask = white_smaller & combi_mask
    # Allow Combi moves if thereby 5 fields are selected in a single row
    combi_mask = np.logical_and.reduce(np.array((
        pos_combi_actions[:, 0, 0] == pos_combi_actions[:, 1, 0],  # same row
        pos_combi_actions[:, 0, 1] < pos_combi_actions[:, 1, 1],  # white smaller than color
        pos_combi_actions[:, 1, 1] == 10,  # is colored field at the end of the row
        np.tile(white_mask, len(pos_color_actions)),  # is white action allowed
        sel_field_row[np.tile(pos_white_actions, (len(pos_color_actions), 1))[:, 0]] >= 3  # 3 slc fields in that row
        ))) | combi_mask
    if error_count >= 1:
        # Disallow combi actions where one row is closed and white closes another row - next move impossible - game over
        combi_mask = combi_mask & (pos_combi_actions[:, 0, 1] != 10)
        # combi_mask = np.where(np.all(pos_combi_actions[:, :, 1] == 10, axis=1), False, combi_mask)
    return pos_combi_actions, combi_mask


class Environment:
    MAX_ERRORS: int = 4
    TOTAL_SCORE_LOOKUP: dict[int, int] = {
        0: 0,
        1: 1,
        2: 3,
        3: 6,
        4: 10,
        5: 15,
        6: 21,
        7: 28,
        8: 36,
        9: 45,
        10: 55,
        11: 66,
        12: 78
    }

    def __init__(self, game_env: GameEnvironment):
        self.sel_fields = np.zeros_like(BOARD).astype(bool)
        self.error_count: int = 0
        self.rows_closed_score = np.zeros((4, 1)).astype(bool)
        self.rows_closed_counter = 0
        self.game_env = game_env

    def reset(self):
        self.sel_fields = np.zeros_like(BOARD).astype(bool)
        self.error_count = 0
        self.rows_closed_score = np.zeros((4, 1)).astype(bool)
        self.rows_closed_counter = 0

    def get_possible_actions(self, white_dr: int,
                             color_dr: np.ndarray):
        pos_white_actions, white_mask = _allowed_white_actions(self.n_slc_row,
                                                               self.dists,
                                                               white_dr,
                                                               self.game_env.CLOSED_ROWS)

        pos_color_actions, color_mask = _allowed_color_actions(self.n_slc_row,
                                                               self.dists,
                                                               color_dr,
                                                               self.game_env.CLOSED_ROWS)

        pos_combi_actions, combi_mask = _allowed_combi_actions(pos_white_actions,
                                                               white_mask,
                                                               pos_color_actions,
                                                               color_mask,
                                                               self.error_count,
                                                               self.n_slc_row)
        pos_moves = [[x] for x in zip(*pos_white_actions)]
        pos_moves += [[x] for x in zip(*pos_color_actions)]
        pos_moves += [[tuple(pos_combi_actions[i, 0, :]), tuple(pos_combi_actions[i, 1, :])]
                      for i in range(pos_combi_actions.shape[0])]
        mask = list(white_mask)
        mask += list(color_mask)
        mask += list(combi_mask)
        return pos_moves, mask

    def get_possible_white_actions(self, white_dr: int):
        return _allowed_white_actions(self.n_slc_row, self.dists, white_dr, self.game_env.CLOSED_ROWS)

    @property
    def n_slc_row(self):
        return np.count_nonzero(self.sel_fields, axis=1)

    @property
    def dists(self):
        return (self.sel_fields.shape[1] - 1 - np.argmax(self.sel_fields[:, ::-1], axis=1) -
                (~np.any(self.sel_fields, axis=1) * self.sel_fields.shape[1]))

    def take_move(self, color: int, num: int):
        """
        Take a move in the environment
        :param color: as int 0: red, 1: yellow, 2: green, 3: blue
        :param num: number to cross out - caution: does not correspond to the index
        :return:
        """
        assert not self.game_env.is_game_over, "Game is over"
        assert 0 <= color <= 4, "color must be between 0 and 4"
        assert 2 <= num <= 12, "num must be between 2 and 12"
        assert self.error_count <= self.MAX_ERRORS, "More errors than allowed - Should be handled by game env"
        assert self.rows_closed_counter <= 2, "Already two rows closed game is over - Should be handled by game env"
        assert color not in self.game_env.CLOSED_ROWS, "Row already closed"
        dist = self.dists[color]
        assert dist < np.where(BOARD == num)[1][color], "Move not allowed"
        if (color in (0, 1) and num == 12) or (color in (2, 3) and num == 2):
            assert np.count_nonzero(self.sel_fields[color, :]) >= 4, "Row not ready to close"
            self.game_env.CLOSED_ROWS.append(color)
            self.rows_closed_score[color, :] = True
            self.rows_closed_counter += 1
        if self.rows_closed_counter >= 2:
            self.game_env.PLAYER_ENDED_GAME = True
        self.sel_fields[list(zip(*np.where(BOARD == num)))[color]] = True

    def return_current_board(self) -> np.ndarray:
        """
        LEGACY
        Returns the possible moves that may be taken by an agent. Puts -1 in fields which are not allowed
        :return:
        """
        warn("Function is outdated", DeprecationWarning)
        block_idx = np.where(np.any(self.sel_fields, axis=1),
                             np.array([11, 11, 11, 11]) - np.argmax(self.sel_fields[:, ::-1],
                                                                    axis=1), 0)[:, np.newaxis]
        x = np.where(np.indices(self.BOARD.shape)[1] >= block_idx, self.BOARD, -1)
        x[:, -1] = np.where(np.count_nonzero(self.sel_fields, axis=1) >= 4, self.BOARD[:, -1], -1)
        x[self.CLOSED_ROWS, :] = -1
        return x

    def compute_total_score(self):
        return (sum(itemgetter(*np.count_nonzero(
            np.hstack((self.sel_fields, self.rows_closed_score)), axis=1))(self.TOTAL_SCORE_LOOKUP))
                - self.error_count * 5)

    def take_error(self):
        self.error_count += 1
        if self.error_count >= self.MAX_ERRORS:
            self.game_env.PLAYER_ENDED_GAME = True


if __name__ == "__main__":
    ge = GameEnvironment()
    env = Environment(ge)
    env.take_move(1, 2)
    env.take_move(1, 3)
    env.take_move(0, 4)
    ge.CLOSED_ROWS.append(0)
    env.take_move(1, 5)
    env.take_move(1, 6)
    _, white_dice, color_dice = throw_dice()
    print(white_dice)
    env.get_possible_actions(white_dice, color_dice)
