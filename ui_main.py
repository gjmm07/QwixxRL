import time
from PyQt5 import QtCore, QtGui, QtWidgets, QtTest
import gui
import numpy as np
from agent import Agent, RealPlayer
from collections import deque
from RL_agent import RLAgent
from typing import Generator
from dice_roll import throw_dice
from environment import GameEnvironment, Environment
from operator import itemgetter
from functools import partial

# pyuic5 /home/linux/helloworld.ui -o helloworld.py


class App(QtWidgets.QMainWindow, gui.Ui_MainWindow):

    LOOKUP_COLORS: dict[str, int] = {"red": 0, "yellow": 1, "green": 2, "blue": 3}

    def __init__(self,
                 players: deque[RealPlayer | Agent | RLAgent],
                 player_order: Generator[RealPlayer | Agent | RealPlayer, None, None],
                 game_env: GameEnvironment,
                 parent=None):

        super().__init__(parent)
        QtCore.QRect(QtCore.QPoint(0, 0), QtCore.QSize(-1, -1))
        self.setupUi(self)
        self.throw_dice_but.clicked.connect(self.throw_dice_callback)
        self.continue_but.clicked.connect(self.continue_callback)
        self.dice_images = np.array([[QtGui.QPixmap(f"dices/{color}{idx}.png").scaled(QtCore.QSize(100, 100))
                                      for idx in range(1, 7)] for color in
                                     ("white", "white", "red", "yellow", "green", "blue")])
        self.game_env: GameEnvironment = game_env

        self.grid: np.ndarray[np.dtype: QtWidgets.QCheckBox] = np.array([[None] * 11] * 4)
        for child in self.gridLayoutWidget.children():
            if isinstance(child, QtWidgets.QGridLayout):
                continue
            color, index = child.objectName().split("_")
            child.clicked.connect(partial(self.get_moves, child, self.LOOKUP_COLORS[color], int(index)))
            self.grid[self.LOOKUP_COLORS[color], int(index) - 2] = child
        self.grid = np.vstack((self.grid[:2, :], self.grid[2:, ::-1]))

        self.error_widgets: list[QtWidgets.QCheckBox] = [self.error_1, self.error_2, self.error_3, self.error_4]
        for error_widget in self.error_widgets:
            error_widget.setEnabled(False)
            error_widget.clicked.connect(partial(self.get_error, error_widget))

        self.sel_moves: list[tuple[int, int] | None] = []

        self.black_img = QtGui.QPixmap(100, 100)
        self.black_img.fill(QtGui.QColor(0, 0, 0))

        self.dice_labels: list[QtWidgets.QLabel] = [self.White_dice_1, self.White_dice_2, self.Red_dice,
                                                    self.Yellow_dice, self.Green_dice, self.Blue_dice]
        self.player_order = player_order
        self.players = players
        self.player: Agent | RealPlayer | None = None

        self.dice_roll: tuple[np.ndarray, int, np.ndarray] = (-np.ones((6, )).astype(int), 0, np.zeros(0, ))
        self.history: deque[np.ndarray] = deque(maxlen=3)

        self.exec_main_move: bool = True
        self.qTimer = QtCore.QTimer()
        self.qTimer.setInterval(100)
        self.qTimer.timeout.connect(self.display_dice)
        self.qTimer.start()
        self.continue_callback()

    def get_moves(self, child: QtWidgets.QCheckBox, color, num):
        sel = child.isChecked()
        if sel:
            self.sel_moves.append((color, num))
        else:
            self.sel_moves.remove((color, num))

    def get_error(self, child: QtWidgets.QCheckBox):
        if child.isChecked():
            self.sel_moves.append(None)
        else:
            self.sel_moves.remove(None)

    def display_dice(self):
        if self.game_env.CLOSED_ROWS:
            self.dice_roll[0][np.array(self.game_env.CLOSED_ROWS) + 2] = -1
        for i, (d_label, dice) in enumerate(zip(self.dice_labels, self.dice_roll[0])):
            if dice == -1:
                d_label.setPixmap(self.black_img)
            else:
                d_label.setPixmap(self.dice_images[i, dice - 1])

    def throw_dice_callback(self):
        """
        If it's the turn of a real player to throw the dice - wait for this callback
        :return:
        """
        self.dice_roll = throw_dice()
        self.load_env()
        self.continue_but.setEnabled(True)
        self.throw_dice_but.setDisabled(True)

    def enable_grid(self, state: bool):
        for widget in self.grid.ravel():
            widget.setEnabled(state)

    def load_env(self):
        # there must be a better way?
        for widget in self.grid[~self.player.env.sel_fields]:
            widget.setChecked(False)
        for widget in self.grid[self.player.env.sel_fields]:
            widget.setChecked(True)
        self.enable_grid(True)
        for row, dist in enumerate(self.player.env.dists):
            if dist >= 0:
                for widget in self.grid[row, :dist + 1]:
                    widget.setDisabled(True)
        for ew in self.error_widgets:
            ew.setChecked(False)
        for i, error_widget in enumerate(self.error_widgets):
            if i < self.player.env.error_count:
                error_widget.setChecked(True)
            if i == self.player.env.error_count:
                error_widget.setEnabled(self.exec_main_move)
            else:
                error_widget.setDisabled(True)

    def real_player_action(self) -> bool:
        if any(x is None for x in self.sel_moves):
            if len(self.sel_moves) > 1:
                return False
            print(self.player.name, "take error")
            self.player.env.take_error()
            return True
        if self.exec_main_move and not (1 <= len(self.sel_moves) <= 2):
            return False
        if not self.exec_main_move and len(self.sel_moves) > 1:
            return False
        for action in self.sel_moves:
            try:
                self.player.env.take_move(*action)
            except AssertionError as ae:
                print(ae)
                return False
        return True

    def unreal_player_action(self) -> bool:
        if self.exec_main_move:
            QtTest.QTest.qWait(1000)
            self.player.do_main_move(*self.dice_roll[1:])
            self.load_env()
            QtTest.QTest.qWait(1000)
        else:
            QtTest.QTest.qWait(1000)
            self.player.downstream_move(self.dice_roll[1])
            self.load_env()
            QtTest.QTest.qWait(1000)
        return True

    def player_do_action(self) -> bool:
        if self.player is None:  # At the beginning
            return True
        if type(self.player).__name__ == "RealPlayer":
            return self.real_player_action()
        else:
            return True

    def continue_callback(self):
        print("________________________")
        if self.game_env.is_game_over:
            return
        if not self.player_do_action():
            print("Not allowed -- Please change your selection")
            return
        self.player = next(self.player_order)
        self.sel_moves.clear()
        if self.player is None:
            self.player = next(self.player_order)
            self.current_player.setText(f"{self.player.name} do main move")
            self.enable_grid(False)
            self.throw_dice_but.setEnabled(True)
            self.continue_but.setDisabled(True)
            self.exec_main_move = True
            if type(self.player).__name__ != "RealPlayer":
                self.throw_dice_callback()
                self.unreal_player_action()
                print("done")
                # self.continue_callback()
        else:
            self.exec_main_move = False
            self.load_env()
            self.current_player.setText(f"{self.player.name} do white move")
            if type(self.player).__name__ != "RealPlayer":
                self.unreal_player_action()
                print("done")
                # self.continue_callback()
        if self.game_env.is_game_over:
            print("GAME OVER")
            self.close()


if __name__ == '__main__':
    pass

