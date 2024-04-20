from ui_main import App
from PyQt5.QtWidgets import QApplication
import sys
from collections import deque
from agent import Agent, RealPlayer
from RL_agent import RLAgent
from typing import Generator
from environment import GameEnvironment, Environment


GAME_ENVIRONMENT = GameEnvironment()
PLAYERS: deque[Agent | RealPlayer | RLAgent] = deque((Agent("Finn", Environment(GAME_ENVIRONMENT)),
                                                      Agent("Luisa", Environment(GAME_ENVIRONMENT))))


def next_player() -> Generator[Agent | RLAgent | RealPlayer, None, None]:
    yield None
    players = PLAYERS.copy()
    while True:
        for player in players:
            yield player
        players.rotate(-1)
        yield None


def main_ui():
    app = QApplication(sys.argv)
    form = App(players=PLAYERS, player_order=next_player(), game_env=GAME_ENVIRONMENT)
    form.show()
    app.exec_()

    for player in PLAYERS:
        player.print_total_score()


if __name__ == "__main__":
    main_ui()

