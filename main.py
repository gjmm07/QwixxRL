from ui_main import App
from PyQt5.QtWidgets import QApplication
import sys
from collections import deque
from stats_palyer import Agent
from RealPlayer import RealPlayer
from MC_RL_agent import RLAgent
from DQL_agent import DQLAgent, Networks
from typing import Generator
from dice_roll import throw_dice
from matplotlib import pyplot as plt
import numpy as np


class PlayerSetup:
    PLAYERS: deque[DQLAgent | RealPlayer] = deque([DQLAgent("Finn"),
                                                   DQLAgent("Luisa")])

    def next_gen(self) -> Generator[tuple[bool, DQLAgent | RealPlayer], None, None]:
        """
        :return:
        """
        players = self.PLAYERS.copy()
        while True:
            for i, player in enumerate(players):
                yield i == 0, player
            yield None, None  # End of round
            players.rotate(-1)

    @staticmethod
    def end_game():
        scores = [p.env.compute_total_score() for p in pg.PLAYERS]
        for p, winner in zip(pg.PLAYERS, [x == max(scores) for x in scores]):
            p.end_game_callback(10 if winner else 0)  # reward the winner extra
            # todo: include amount of moves made as reward?
        PlayerSetup.PLAYERS.rotate(-1)
        return scores

    @staticmethod
    def end_round():
        for p in pg.PLAYERS:
            p.end_round_callback()

    @staticmethod
    def start_round():
        for p in pg.PLAYERS:
            p.start_round_callback()


pg = PlayerSetup()
nets = Networks((51, ), 45, (32, 8, 32))


def sim_main(n_games: int = 1):
    epsilon = 1
    avg_scores = []
    for game in range(n_games):
        for first, player in pg.next_gen():
            if player is None:
                pg.end_round()
                continue
            if player.is_game_over:
                break
            if first:
                pg.start_round()
                dr = throw_dice()
                player.do_main_move(dice_roll=dr, net=nets, epsilon=epsilon)
            else:
                player.downstream_move(dr)
        scores = pg.end_game()
        avg_scores.append(np.mean(scores))
        nets.train(DQLAgent.replay_memory)
        if not game % 100:
            print(f"{round(epsilon, 2)} \t {avg_scores[-1]}")
            nets.copy_weights()
        epsilon -= 1 / n_games
    plt.plot(avg_scores)
    plt.show()


def main():
    if any(type(p) is RealPlayer for p in pg.PLAYERS):
        app = QApplication(sys.argv)
        form = App(player_setup=pg)
        form.show()
        app.exec_()
    else:
        sim_main()


if __name__ == "__main__":
    main()
