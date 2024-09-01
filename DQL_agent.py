import keras.optimizers
from _Player import _Player
import numpy as np
from environment import ExtraType
from dice_roll import get_moves, throw_dice
from collections import deque
from dataclasses import dataclass
import typing
import random
from keras import Sequential
from keras.layers import Dense, Input
import tensorflow as tf


@dataclass
class Memory:
    map: np.ndarray
    error_count: int
    dice_roll: np.ndarray

    next_map: np.ndarray = None
    next_allowed: list[bool] = None
    next_error_count: int = None
    next_dice_roll: np.ndarray = None

    action: int = None
    reward: int = None
    terminate: bool = None

    @property
    def state(self):
        return np.hstack((self.map.ravel(), np.atleast_1d(self.error_count), self.dice_roll))

    @property
    def next_state(self):
        return np.hstack((self.next_map.ravel(), np.atleast_1d(self.next_error_count), self.next_dice_roll / 6))


class Networks:
    loss_fn = tf.keras.losses.MeanSquaredError()

    def __init__(self, input_shape: tuple[int], output: int, hidden_layer_neurons: tuple[int, ...]):
        self._target_net: Sequential = Sequential()
        self._target_net.add(Input(shape=input_shape))
        for hl in hidden_layer_neurons:
            self._target_net.add(Dense(hl, activation="relu"))
        self._target_net.add(Dense(output, activation="linear"))

        self._policy_net: Sequential = tf.keras.models.clone_model(self._target_net)
        self._n_output = output
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-5)

    def train(self, replay_memory: typing.Sequence[Memory]):
        batch_size = 150
        if len(replay_memory) < batch_size:
            batch_size = len(replay_memory)
        batch = random.sample(replay_memory, batch_size)
        states, next_states, next_allowed, actions, rewards, term = zip(
            *((r.state, r.next_state, r.next_allowed, r.action, r.reward, r.terminate) for r in batch))
        q_s_a_prime = np.max(
            self._target_net(
                np.vstack(next_states), training=True), axis=1, where=next_allowed, initial=-1)
        q_s_a_target = np.where(term, rewards, rewards + 0.9 * q_s_a_prime)
        q_s_a_target = tf.convert_to_tensor(q_s_a_target, dtype="float32")
        with tf.GradientTape() as tape:
            q_s_a = tf.reduce_sum(
                self._policy_net(np.vstack(states)) * tf.one_hot(actions, self._n_output), axis=1)
            loss = Networks.loss_fn(q_s_a_target, q_s_a)
        grads = tape.gradient(loss, self._policy_net.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self._policy_net.trainable_weights))

    def copy_weights(self):
        self._target_net.set_weights(self._policy_net.get_weights())

    def get_best_action(self, state, allowed: tuple[bool]):
        pred = self._policy_net.predict(state, verbose=False)
        pred = pred[0]
        valid_idx = np.where(allowed)[0]
        return valid_idx[np.argmax(pred[valid_idx])]


class DQLAgent(_Player):
    replay_memory: deque[Memory] = deque([], maxlen=1000)
    games: int = 0

    def __init__(self, name: str):
        super().__init__(name)
        self._score: int = 0
        self._type: typing.Literal["main", "downstream"] = "main"
        self._first_round = True

        self._memory: Memory = Memory(np.empty((1, )), -100, np.empty((1, )))

    def do_main_move(self, dice_roll: np.ndarray, net: Networks, epsilon: float):
        """
        state: 44 bool for each field + error count + 6 dices (2 white (sorted) 4 color) = 51
        (4, 2) single color actions, 4 white actions, combo moves and error = 45 possible moves
        in combination with 2⁴⁴ possible states this is impossible to model in a q-table
        """
        self._type = "main"
        dice_roll[0:2].sort()
        self._memory.map = self.env.sel_fields.astype(int)
        self._memory.error_count = self.env.error_count / 4
        self._memory.dice_roll = dice_roll / 6
        white_dr, color_dr = get_moves(dice_roll)
        moves, allowed = self.env.get_possible_actions(white_dr, color_dr)
        allowed += [True]
        moves += [ExtraType.error]
        if random.random() < epsilon:
            action_idx = self._select_random_action(allowed)
        else:
            action_idx = net.get_best_action(np.atleast_2d(self._memory.state), allowed)
        self._memory.action = action_idx
        action = moves[action_idx]
        super().take_action(action)
        return action

    def downstream_move(self, dice_roll: np.ndarray):
        """
        state: 44 bool for each field + white dice sum = 45 states
        actions: 4 white actions and blank moves = 5 possible actions
        """
        self._type = "downstream"
        return ExtraType.blank
        # white_dr = np.sum(dice_roll[:2])
        # self._state = np.append(self.env.sel_fields.ravel(), white_dr / 12)
        # moves, allowed = self.env.get_possible_white_actions(white_dr)
        # allowed += [True]
        # moves += [ExtraType.blank]
        # if random.random() < self._epsilon:
        #     action_idx = self._select_random_action(allowed)
        # else:
        #     action_idx = downstream_net.get_best_action(np.atleast_2d(self._state), allowed)
        # self._action = action_idx
        # action = moves[action_idx]
        # super().take_action(action)
        # return action

    def _select_random_action(self, allowed):
        return np.random.choice(np.where(allowed)[0])

    def _add_to_replay(self, reward: int, terminate: bool = False):
        if self._type == "downstream":
            return
        self._memory.next_map = self.env.sel_fields.astype(int)
        self._memory.next_error_count = self.env.error_count / 4
        dice_roll = throw_dice()
        dice_roll[:2].sort()
        _, next_allowed = self.env.get_possible_actions(*get_moves(dice_roll))
        next_allowed += (True, )
        self._memory.next_allowed = next_allowed
        self._memory.next_dice_roll = dice_roll
        self._memory.reward = reward
        self._memory.terminate = terminate
        DQLAgent.replay_memory.append(self._memory)

    def end_round_callback(self):
        return

    def start_round_callback(self, *args, **kwargs):
        if self._first_round:
            self._first_round = False
            return
        new_score = self.env.compute_total_score()
        reward = (new_score - self._score) + (40 - np.sum(self.env.dists)) / 40  # rewards score + clickable fields
        # todo: include amount of moves made as reward?
        self._add_to_replay(reward)
        self._score = new_score

    def end_game_callback(self, end_game_reward: int):
        reward = end_game_reward
        self._add_to_replay(reward, True)
        self.env.reset()
        self._first_round = True
        return self.env.compute_total_score()

    @property
    def is_real(self):
        return False
