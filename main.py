from ui_main import App
from PyQt5.QtWidgets import QApplication
import sys
from collections import deque
from agent import Agent
from RealPlayer import RealPlayer
from RL_agent import RLAgent
from typing import Generator
from dice_roll import throw_dice


PLAYERS: deque[Agent | RLAgent | RealPlayer] = deque([RLAgent("Finn"),
                                                      RLAgent("Luisa")])


def next_player() -> Generator[tuple[bool, Agent | RLAgent | RealPlayer], None, None]:
    players = PLAYERS.copy()
    while True:
        for i, player in enumerate(players):
            yield i == 0, player
        players.rotate(-1)


def train_rl_agent(n_games: int):
    for game in range(n_games):
        player_gen: Generator = next_player()
        for is_first, player in player_gen:
            if is_first:
                dice_roll = throw_dice()
                player.do_main_move(*dice_roll[1:])
                if player.env.is_game_over:
                    break
            else:
                player.downstream_move(dice_roll[1])
                if player.env.is_game_over:
                    break
        scores = _compute_score()
        _execute_backpropagation(scores)


def _compute_score():
    total_score = []
    for player in PLAYERS:
        total_score.append(player.end_game_callback())
    return [int(ts == max(total_score)) for ts in total_score]


def _execute_backpropagation(scores: list[int], reset_players: bool = True):
    for score, player in zip(scores, PLAYERS):
        print(player.name)
        player.backpropagation(score)
        if reset_players:
            player.reset_callback()


def main_ui():
    app = QApplication(sys.argv)
    form = App(player_order=next_player())
    form.show()
    app.exec_()

    total_score = _compute_score()
    _execute_backpropagation(total_score)


if __name__ == "__main__":
    train_rl_agent(50)

