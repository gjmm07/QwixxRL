from ui_main import App
from PyQt5.QtWidgets import QApplication
import sys
from collections import deque
from RealPlayer import RealPlayer
from DQL_agent import DQLAgent, Networks, Memory, CNNNetworks
from typing import Generator
from dice_roll import throw_dice
from matplotlib import pyplot as plt
from scipy.ndimage import uniform_filter1d
import matplotlib.colors as mplc
import argparse
from itertools import zip_longest

##########################
####### PARAMETERS #######
##########################

REPLAY_MEMORY: deque[Memory] = deque([], maxlen=500)

# small_nets = Networks.for_training((16, ), 14, (64, 16, 8, 64), 1e-3, 0.99)
# big_nets = Networks.for_training((52, ), 14, (128, 64, 16, 64, 128), 1e-3, 0.99)
# nets: CNNNetworks | Networks = CNNNetworks()
# small_nets = Networks.for_gameplay("models/16_model.keras")
NETS = {"small": Networks.for_training((16,), 14, (64, 16, 8, 64), 1e-3, 0.99),
         "big": Networks.for_training((52, ), 14, (128, 64, 16, 64, 128), 1e-3, 0.99)}


class PlayerSetup:
    # PLAYERS: deque[DQLAgent | RealPlayer] = deque()

    def __init__(self, player: deque[DQLAgent | RealPlayer]):
        self.PLAYERS = player

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

    # @staticmethod
    def end_game(self, print_score: bool = False):
        scores = [p.env.compute_total_score() for p in self.PLAYERS]
        for p, score in zip(self.PLAYERS, scores):
            is_winner = score == max(scores)
            p.end_game_callback(2000 if is_winner else -2000)  # reward the winner extra
            # todo: include amount of moves made as reward?
            if print_score:
                print("{name} {win_lose} with {points}".format(name=p.name,
                                                               win_lose="wins" if is_winner else "looses",
                                                               points=score))
        self.PLAYERS.rotate(-1)
        return scores

    # @staticmethod
    def end_round(self):
        for p in self.PLAYERS:
            p.end_round_callback()

    # @staticmethod
    def start_round(self):
        for p in self.PLAYERS:
            p.start_round_callback()


def train(n_games: int, nets: set):
    epsilon = 1
    avg_scores = []
    # nets = [small_nets, big_nets]
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
            net.train(REPLAY_MEMORY)
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


def play():
    app = QApplication(sys.argv)
    form = App(player_setup=pg)
    form.show()
    app.exec_()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type", choices=["train", "play"])
    parser.add_argument("-s", "--simp", action="append")
    parser.add_argument("-n", "--net", action="append")
    parser.add_argument("-r", "--realp", action="append")
    parser.add_argument("-e", "--epochs", type=int)
    # todo: There is still quite a bit to do here,
    #  e.g. keep order real player sim player,
    #  in play mode let simp have a model,
    #  do checks: enough nets for simp,  epochs define for training
    args = parser.parse_args()
    player_nets = [NETS[x] for x in args.net]
    pg = PlayerSetup(deque([RealPlayer(name) for name in args.realp] +
                           [DQLAgent(name, REPLAY_MEMORY, net) for name, net in zip_longest(args.simp, player_nets)]))
    if args.type == "train":
        train(args.epochs, set(player_nets))
    elif args.type == "play":
        play()
