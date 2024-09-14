from ui_main import App
from PyQt5.QtWidgets import QApplication
import sys
from collections import deque
from stats_palyer import Agent
from RealPlayer import RealPlayer
from MC_RL_agent import RLAgent
from DQL_agent import DQLAgent, Networks, Memory, CNNNetworks
from typing import Generator
from dice_roll import throw_dice
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib.colors as mplc

##########################
####### PARAMETERS #######
##########################

replay_memory: deque[Memory] = deque([], maxlen=500)

# small_nets = Networks.for_training((16, ), 14, (64, 16, 8, 64), 1e-3, 0.99)
# big_nets = Networks.for_training((52, ), 14, (128, 64, 16, 64, 128), 1e-3, 0.99)
# nets: CNNNetworks | Networks = CNNNetworks()
small_nets = Networks.for_gameplay("models/16_model.keras")


class PlayerSetup:
    PLAYERS: deque[DQLAgent | RealPlayer] = deque([
        RealPlayer("Finn"),
        DQLAgent("Luisa", replay_memory, model=small_nets),
        # DQLAgent("Finn", replay_memory, model=big_nets)
    ])

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
    def end_game(print_score: bool = False):
        scores = [p.env.compute_total_score() for p in PlayerSetup.PLAYERS]
        for p, score in zip(PlayerSetup.PLAYERS, scores):
            is_winner = score == max(scores)
            p.end_game_callback(2000 if is_winner else -2000)  # reward the winner extra
            # todo: include amount of moves made as reward?
            if print_score:
                print("{name} {win_lose} with {points}".format(name=p.name,
                                                               win_lose="wins" if is_winner else "looses",
                                                               points=score))
        PlayerSetup.PLAYERS.rotate(-1)
        return scores

    @staticmethod
    def end_round():
        for p in PlayerSetup.PLAYERS:
            p.end_round_callback()

    @staticmethod
    def start_round():
        for p in PlayerSetup.PLAYERS:
            p.start_round_callback()


pg = PlayerSetup()


def sim_main(n_games: int = 10_000):
    epsilon = 1
    avg_scores = []
    nets = [small_nets, big_nets]
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
                player.do_main_move(dice_roll=dr, epsilon=epsilon)
            else:
                player.downstream_move(dr, epsilon=epsilon)
        scores = pg.end_game()
        avg_scores.append(scores)
        for net in nets:
            net.train(replay_memory)
        if not game % 50:
            print(f"{round(epsilon, 2)} \t {avg_scores[-1]}")
            for net in nets:
                net.copy_weights()
        epsilon -= 1 / n_games
        epsilon = max((0.01, epsilon))
    for l, color in zip(zip(*avg_scores), mplc.TABLEAU_COLORS.keys()):
        plt.plot(l, alpha=0.3, color=color, linewidth=0.1)
        plt.plot(uniform_filter1d(l, 50), color=color)
    plt.show()
    for net in nets:
        net.save()


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
