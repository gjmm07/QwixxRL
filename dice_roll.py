import numpy as np


def throw_dice() -> np.ndarray:
    dices = np.random.randint(1, 7, (6,))
    return dices


def get_moves(dices: np.ndarray) -> tuple[int, np.ndarray]:
    return np.sum(dices[:2]), dices[np.newaxis, :2] + dices[2:, np.newaxis]

