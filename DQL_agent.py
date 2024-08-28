from Player import Player
import numpy as np
from environment import ExtraType
from dice_roll import get_moves, throw_dice
from collections import deque
from dataclasses import dataclass
import typing
import random
from keras import Sequential
from tensorflow.keras.layers import Dense, Layer
import tensorflow as tf


@dataclass
class ReplayMemory:
    state: np.ndarray
    next_state: np.ndarray
    action: int
    reward: int
    terminate: bool


class DQLAgent(Player):
    main_replay: deque[ReplayMemory] = deque([], maxlen=1000)
    downstream_replay: deque[ReplayMemory] = deque([], maxlen=1000)

    target_net_main: Sequential = Sequential()
    target_net_main.add(Dense(32, activation="relu"))
    target_net_main.add(Dense(8, activation="relu"))
    target_net_main.add(Dense(32, activation="relu"))
    target_net_main.add(Dense(45, activation="linear"))

    policy_net_main: Sequential = tf.keras.models.clone_model(target_net_main)

    def __init__(self, name: str):
        super().__init__(name)
        self._score: int = 0
        self._state = None
        self._next_state = None
        self._action = None
        self._type: typing.Literal["main", "downstream"] = "main"

    def do_main_move(self, dice_roll: np.ndarray):
        """
        state: 44 bool for each field + 6 dices (2 white (sorted) 4 color) = 50
        (4, 2) single color actions, 4 white actions, combo moves and error = 45 possible moves
        in combination with 2⁴⁴ possible states this is impossible to model in a q-table
        """
        self._type = "main"
        dice_roll[0:2].sort()
        self._state = np.concatenate((self.env.sel_fields.ravel().astype(int), dice_roll / 6), axis=0)
        white_dr, color_dr = get_moves(dice_roll)
        moves, allowed = self.env.get_possible_actions(white_dr, color_dr)
        allowed += [True]
        moves += [ExtraType.error]
        action_idx = np.random.choice(np.where(allowed)[0])
        self._action = action_idx
        action = moves[action_idx]
        super().take_action(action)
        return action

    def downstream_move(self, dice_roll: np.ndarray):
        """
        state: 44 bool for each field + white dice sum = 45 states
        actions: 4 white actions and blank moves = 5 possible actions
        """
        self._type = "downstream"
        white_dr = np.sum(dice_roll[:2])
        self._state = np.append(self.env.sel_fields.ravel(), white_dr / 12)
        moves, allowed = self.env.get_possible_white_actions(white_dr)
        allowed += [True]
        moves += [ExtraType.blank]
        action_idx = np.random.choice(np.where(allowed)[0])
        self._action = action_idx
        action = moves[action_idx]
        super().take_action(action)
        return action

    def _add_to_replay(self, reward: int, terminate: bool = False):
        if self._type == "main":
            dice_roll = throw_dice()
            dice_roll[:2].sort()
            dice_roll = dice_roll / 6
            next_state = np.concatenate((self.env.sel_fields.ravel().astype(int),
                                        dice_roll))
            DQLAgent.main_replay.append(
                ReplayMemory(state=self._state,
                             next_state=next_state,
                             action=self._action,
                             reward=reward,
                             terminate=terminate))
        else:
            dice_roll = np.sum(throw_dice()[:2]) / 12
            next_state = np.append(self.env.sel_fields.ravel().astype(int), dice_roll)
            DQLAgent.downstream_replay.append(
                ReplayMemory(state=self._state,
                             next_state=next_state,
                             action=self._action,
                             reward=reward,
                             terminate=terminate))


    def _train(self):
        replay = random.sample(DQLAgent.main_replay, 10)
        states, next_states, actions, rewards, term = zip(
            *((r.state, r.next_state, r.action, r.reward, r.terminate) for r in replay))
        q_s_a_prime = np.max(DQLAgent.target_net_main.predict(np.vstack(next_states)), axis=1)
        q_s_a_target = np.where(term, rewards, rewards + 0.99 * q_s_a_prime)
        print(q_s_a_target)

    def _copy_weights(self):
        DQLAgent.target_net_main.set_weights(DQLAgent.policy_net_main.get_weights())

    def end_round_callback(self):
        new_score = self.env.compute_total_score()
        reward = new_score - self._score
        self._add_to_replay(reward)
        self._score = new_score

    def end_game_callback(self, end_game_reward: int):
        reward = end_game_reward
        self._add_to_replay(reward, True)  # final state -- what to do?
        self.env.reset()
        self._train()

    @property
    def is_real(self):
        return False
