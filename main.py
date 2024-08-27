from ui_main import App
from PyQt5.QtWidgets import QApplication
import sys
from collections import deque
from stats_palyer import Agent
from RealPlayer import RealPlayer
from MC_RL_agent import RLAgent
from DQL_agent import DQLAgent
from typing import Generator
from dice_roll import throw_dice


class PlayerGenerator:

    PLAYERS: deque[DQLAgent | RLAgent | RealPlayer] = deque([DQLAgent("Finn"),
                                                             DQLAgent("Luisa")])

    def next_player(self) -> Generator[tuple[bool, Agent | RLAgent | RealPlayer], None, None]:
        """
        :return:
        """
        players = self.PLAYERS.copy()
        while True:
            for i, player in enumerate(players):
                yield i == 0, player
            yield None, None  # End of round
            players.rotate(-1)

    def new_round(self):
        PlayerGenerator.PLAYERS.rotate(-1)


pg = PlayerGenerator()


def sim_main(n_games: int = 10):
    for game in range(n_games):
        for first, player in pg.next_player():
            if player is None:
                for p in pg.PLAYERS:
                    p.end_round_callback()
                continue
            if player.is_game_over:
                break
            if first:
                _, wdr, cdr = throw_dice()
                player.do_main_move(wdr, cdr)
            else:
                player.downstream_move(wdr)
        for p in pg.PLAYERS:
            p.end_game_callback()
        pg.new_round()


def main():
    if any(type(p) == RealPlayer for p in pg.PLAYERS):
        app = QApplication(sys.argv)
        form = App(player_order=pg.next_player())
        form.show()
        app.exec_()
    else:
        sim_main()


if __name__ == "__main__":
    main()



