from __future__ import annotations
from PyQt5 import QtCore, QtGui, QtWidgets, QtTest
import gui
import numpy as np
from _Player import _Player
from collections import deque
from dice_roll import throw_dice
from functools import partial
from environment import BOARD, ExtraType
import itertools
from typing import Literal, TYPE_CHECKING
if TYPE_CHECKING:
    from main import PlayerSetup

# pyuic5 /home/linux/helloworld.ui -o helloworld.py


class App(QtWidgets.QMainWindow, gui.Ui_MainWindow):

    LOOKUP_COLORS: dict[str, int] = {"red": 0, "yellow": 1, "green": 2, "blue": 3}

    def __init__(self,
                 player_setup: PlayerSetup,
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
        self.player_gen = player_setup.next_gen()
        self._player_setup = player_setup
        self.player: _Player | None = None

        self.dice_roll: tuple[np.ndarray] = -np.ones((6, )).astype(int)
        self.history: deque[np.ndarray] = deque(maxlen=3)

        self.exec_main_move: bool = True
        self.qTimer = QtCore.QTimer()
        self.qTimer.setInterval(100)
        self.qTimer.timeout.connect(self.main)
        self.qTimer.start()

        self._lock: bool = False
        self._is_first_player: bool = False
        self._disabled_rows: list[int] = []
        self._new_round: bool = True
        self._unreal_player_gen = self.unreal_player_action()
        self.display_dice()

    @staticmethod
    def blinking_button(but, color: Literal["white", "red"]):
        orig_color = but.palette().color(but.backgroundRole()).name()
        for _ in range(10):
            but.setStyleSheet(f"background-color: {color}")
            yield
            but.setStyleSheet(f"background-color: {orig_color}")
            yield

    @staticmethod
    def _wait(x: int or float):
        x = int(x)
        for _ in range(x):
            yield

    def unreal_player_action(self, pace: float = 0.5):
        """

        :param pace:
        :return:
        """
        while True:
            yield from self._wait(5 / pace)
            if self._is_first_player:
                self.dice_roll = throw_dice()
            self.display_dice()
            yield from self._wait(15 / pace)
            if self._is_first_player:
                actions = self.player.do_main_move(self.dice_roll, epsilon=0)
            else:
                actions = self.player.downstream_move(self.dice_roll, epsilon=0)
            yield from self._wait(10 / pace)
            if isinstance(actions, ExtraType):
                if self._is_first_player:
                    self.error_widgets[self.player.env.error_count - 1].setChecked(True)
                    yield from self.blinking_button(self.error_widgets[self.player.env.error_count - 1], "red")
            else:
                for a in actions:
                    self.grid[a[0]][a[1]].setChecked(True)
                    yield from self.blinking_button(self.grid[a[0]][a[1]], "white")
            yield from self._wait(20 / pace)
            self._lock = False
            yield

    def main(self):
        if self._lock:
            if not self.player.is_real:
                next(self._unreal_player_gen)
            return
        self._is_first_player, self.player = next(self.player_gen)
        if self.player is None:
            self._player_setup.end_round()
            return
        if self._is_first_player:
            self._player_setup.start_round()
            self._new_round = True
            self._lock = False
        if self._new_round:
            self._disabled_rows = self.player.env.closed_rows.copy()
            self._new_round = False
        self.current_player.setText("" if self.player is None else self.player.name)
        self.setup_player()
        if self.player.env.is_game_over:
            # todo: Properly end game and reset
            self._player_setup.end_game(True)
            self.qTimer.stop()
            self.close()
        self._lock = True

    def continue_callback(self):
        if self.player is None or not self.player.is_real:
            return
        if not self.sel_moves:
            self.player.take_action(ExtraType.blank)
        elif ExtraType.error in self.sel_moves:
            self.player.take_action(ExtraType.error)
        else:
            self.player.take_action(self.sel_moves)
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
        j = 0
        k = 0
        d = self.player.env.dists[j]
        for i, (wid, selected) in enumerate(zip(itertools.chain.from_iterable(self.grid),
                                                self.player.env.sel_fields.ravel())):
            if not i % 11 and i != 0:
                j += 1
                d = self.player.env.dists[j]
                k += 11
            wid.setEnabled(d + k + 1 <= i)
            wid.setChecked(selected)

    def _load_error_grid(self):
        for i, ew in enumerate(self.error_widgets):
            ew.setEnabled(i < self.player.env.error_count + 1)
            ew.setChecked(i < self.player.env.error_count)

    def setup_player(self):
        if self.player is None:
            return
        self.throw_dice_but.setDisabled(not self._is_first_player)
        self.continue_but.setDisabled(self._is_first_player)
        self._load_grid()
        self._load_error_grid()

    def get_moves(self, child: QtWidgets.QCheckBox, color, num):
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
        for i, (d_label, dice) in enumerate(zip(self.dice_labels, self.dice_roll)):
            if dice == -1 or i in (d + 2 for d in self.player.env.closed_rows):
                d_label.setPixmap(self.black_img)
            else:
                d_label.setPixmap(self.dice_images[i, dice - 1])


if __name__ == '__main__':
    pass

