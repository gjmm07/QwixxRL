import numpy as np


def throw_dice() -> tuple[np.ndarray, int, np.ndarray]:
    dices = np.random.randint(1, 7, (6,))
    print(np.sum(dices[:2]), dices[np.newaxis, :2] + dices[2:, np.newaxis])
    return dices, np.sum(dices[:2]), dices[np.newaxis, :2] + dices[2:, np.newaxis]

