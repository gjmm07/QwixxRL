import time
from PyQt5 import QtCore, QtGui, QtWidgets, QtTest
import gui
import numpy as np
from agent import Agent
from RealPlayer import RealPlayer
from collections import deque
from RL_agent import RLAgent
from typing import Generator
from dice_roll import throw_dice
from functools import partial
from environment import BOARD, ExtraType
import itertools

# pyuic5 /home/linux/helloworld.ui -o helloworld.py


class App(QtWidgets.QMainWindow, gui.Ui_MainWindow):

    LOOKUP_COLORS: dict[str, int] = {"red": 0, "yellow": 1, "green": 2, "blue": 3}

    def __init__(self,
                 player_order: Generator[tuple[bool, RealPlayer | Agent | RLAgent], None, None],
                 parent=None):

        super().__init__(parent)
        QtCore.QRect(QtCore.QPoint(0, 0), QtCore.QSize(-1, -1))
        self.setupUi(self)
        self.throw_dice_but.clicked.connect(self.throw_dice_callback)
        self.continue_but.clicked.connect(self.continue_callback)
        self.dice_images = np.array([[QtGui.QPixmap(f"dices/{color}{idx}.png").scaled(QtCore.QSize(100, 100))
                                      for idx in range(1, 7)] for color in
                                     ("white", "white", "red", "yellow", "green", "blue")])

        grid = [[], [], [], []]
        for color in ("red", "yellow", "green", "blue"):
            for i, num in enumerate(reversed(range(2, 13)) if color in ("green", "blue") else range(2, 13)):
                wid = self.__getattribute__(f"{color}_{num}")
                grid[self.LOOKUP_COLORS[color]].append(wid)
                wid.clicked.connect(partial(self.get_moves,
                                            wid,
                                            self.LOOKUP_COLORS[color],
                                            i))
        self.grid: list[list[QtWidgets.QCheckBox]] = grid

        self.error_widgets: list[QtWidgets.QCheckBox] = [self.error_1, self.error_2, self.error_3, self.error_4]
        for error_widget in self.error_widgets:
            error_widget.setEnabled(False)
            error_widget.clicked.connect(partial(self.get_error, error_widget))

        self.sel_moves: list[tuple[int, int] | ExtraType] = []

        self.black_img = QtGui.QPixmap(100, 100)
        self.black_img.fill(QtGui.QColor(0, 0, 0))

        self.dice_labels: list[QtWidgets.QLabel] = [self.White_dice_1, self.White_dice_2, self.Red_dice,
                                                    self.Yellow_dice, self.Green_dice, self.Blue_dice]
        self.player_order = player_order
        self.player: Agent | RealPlayer | None = None

        self.dice_roll: tuple[np.ndarray, int, np.ndarray] = (-np.ones((6, )).astype(int), 0, np.zeros(0, ))
        self.history: deque[np.ndarray] = deque(maxlen=3)

        self.exec_main_move: bool = True
        self.qTimer = QtCore.QTimer()
        self.qTimer.setInterval(100)
        self.qTimer.timeout.connect(self.main)
        self.qTimer.start()

        self._lock: bool = False
        self._first_player: bool = False
        self._unreal_player_gen = self.unreal_player_action()
        self.display_dice()

    def unreal_player_action(self):
        while True:
            for _ in range(10):
                yield
            self.dice_roll = throw_dice()
            self.display_dice()
            for _ in range(30):
                yield
            if self._first_player:
                actions = self.player.do_main_move(*self.dice_roll[1:])
            else:
                actions = self.player.downstream_move(self.dice_roll[1])
            for _ in range(5):
                yield
            for a in actions:
                orig_color = self.grid[a[0]][a[1]].palette().color(self.grid[a[0]][a[1]].backgroundRole()).name()
                self.grid[a[0]][a[1]].setChecked(True)
                for _ in range(10):
                    self.grid[a[0]][a[1]].setStyleSheet("background-color: white")
                    yield
                    self.grid[a[0]][a[1]].setStyleSheet(f"background-color: {orig_color}")
                    yield
            for _ in range(80):
                yield
            self._lock = False
            yield

    def main(self):
        if not self._lock:
            self._first_player, self.player = next(self.player_order)
            self.current_player.setText("" if self.player is None else self.player.name)
            self.setup_player()
            self._lock = True
        if self.player is None:
            print("New round")
            self._lock = False
            return
        if not self.player.is_real:
            next(self._unreal_player_gen)

    def continue_callback(self):
        if self.player is None or not self.player.is_real:
            return
        try:
            self.player.take_action(self.sel_moves, self._first_player, *self.dice_roll[1:])
        except AssertionError as ae:
            print(ae)
            return
        self.sel_moves = []
        self._lock = False

    def throw_dice_callback(self):
        if self.player is None or not self.player.is_real:
            return
        self.dice_roll = throw_dice()
        self.throw_dice_but.setDisabled(True)
        self.continue_but.setDisabled(False)
        self.display_dice()

    def _load_grid(self):
        for wid, selected in zip(itertools.chain.from_iterable(self.grid), self.player.env.sel_fields.ravel()):
            wid.setChecked(selected)

    def _load_error_grid(self):
        for i, ew in enumerate(self.error_widgets):
            ew.setEnabled(i < self.player.env.error_count + 1)
            ew.setChecked(i < self.player.env.error_count)

    def setup_player(self):
        if self.player is None:
            return
        self.throw_dice_but.setDisabled(not self._first_player)
        self.continue_but.setDisabled(self._first_player)
        self._load_grid()
        self._load_error_grid()

    def get_moves(self, child: QtWidgets.QCheckBox, color, num):
        print(color, num)
        sel = child.isChecked()
        if sel:
            self.sel_moves.append((color, num))
        else:
            self.sel_moves.remove((color, num))

    def get_error(self, child: QtWidgets.QCheckBox):
        if child.isChecked():
            self.sel_moves.append(ExtraType.error)
        else:
            self.sel_moves.remove(ExtraType.error)

    def display_dice(self):
        for i, (d_label, dice) in enumerate(zip(self.dice_labels, self.dice_roll[0])):
            if dice == -1:
                d_label.setPixmap(self.black_img)
            else:
                d_label.setPixmap(self.dice_images[i, dice - 1])


if __name__ == '__main__':
    pass

