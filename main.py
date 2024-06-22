from ui_main import App
from PyQt5.QtWidgets import QApplication
import sys
from collections import deque
from agent import Agent
from RealPlayer import RealPlayer
from RL_agent import RLAgent
from typing import Generator


PLAYERS: deque[Agent | RLAgent | RealPlayer] = deque([RealPlayer("Finn"),
                                                      RLAgent("Luisa")])


def next_player() -> Generator[tuple[bool, Agent | RLAgent | RealPlayer], None, None]:
    yield False, None
    players = PLAYERS.copy()
    while True:
        for i, player in enumerate(players):
            yield i == 0, player
        players.rotate(-1)
        yield False, None


def main_ui():
    app = QApplication(sys.argv)
    form = App(player_order=next_player())
    form.show()
    app.exec_()

    for player in PLAYERS:
        player.print_total_score()


if __name__ == "__main__":
    main_ui()

