from __future__ import annotations
import keras.optimizers
from _Player import _Player
import numpy as np
from environment import ExtraType, dists, n_slc_row
from dice_roll import get_moves, throw_dice
from collections import deque
from dataclasses import dataclass
import typing
import random
from keras import Sequential, Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Concatenate
# from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Concatenate
import tensorflow as tf
import copy


def sample_replay_memory(replay_memory: typing.Sequence[Memory], model_type: Networks | CNNNetworks):
    batch_size = 150
    if len(replay_memory) < batch_size:
        batch_size = len(replay_memory)
    batch = random.sample(replay_memory, batch_size)
    return zip(
        *((r.get_state(model_type), r.get_next_state(model_type), r.next_allowed, r.action, r.reward, r.terminate)
          for r in batch))


class Networks:
    loss_fn = tf.keras.losses.MeanSquaredError()

    def __init__(self,
                 input_shape: tuple[int],
                 output: int,
                 hidden_layer_neurons: tuple[int, ...]):

        self._target_net: Sequential = Sequential()
        self._target_net.add(Input(shape=input_shape))
        for hl in hidden_layer_neurons:
            self._target_net.add(Dense(hl, activation="relu"))
        self._target_net.add(Dense(output, activation="linear"))

        self._policy_net: Sequential = tf.keras.models.clone_model(self._target_net)

        self._target_net.compile(keras.optimizers.Adam(learning_rate=1e-4), loss=self.loss_fn)
        self._policy_net.compile(keras.optimizers.Adam(learning_rate=1e-4), loss=self.loss_fn)
        self._n_output = output
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-5)

    def train(self, replay_memory: typing.Sequence[Memory]):
        states, next_states, next_allowed, actions, rewards, term = sample_replay_memory(replay_memory, model_type=self)
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

    def train2(self, replay_memory: typing.Sequence[Memory]):
        states, next_states, next_allowed, actions, rewards, term = sample_replay_memory(replay_memory, model_type=self)
        states = np.array(states)
        qsa_prime = np.max(
            self._target_net.predict(np.array(next_states), verbose=0), axis=1, where=next_allowed, initial=-1)
        qsa = self._policy_net.predict(states, verbose=0)
        np.put_along_axis(
            qsa, np.atleast_2d(actions).T, np.atleast_2d(np.where(term, rewards, rewards + 0.99 * qsa_prime)).T, axis=1)
        x = self._policy_net.train_on_batch(states, qsa)
        print(x)

    def copy_weights(self):
        self._target_net.set_weights(self._policy_net.get_weights())

    def get_best_action(self, state: np.ndarray, allowed: tuple[bool]):
        state = np.atleast_2d(state)
        pred = self._policy_net.predict(state, verbose=False)
        pred = pred[0]
        valid_idx = np.where(allowed)[0]
        return valid_idx[np.argmax(pred[valid_idx])]

    @property
    def input_shape(self):
        return self._target_net.input_shape


class CNNNetworks:
    loss_fn = tf.keras.losses.MeanSquaredError()

    def __init__(self):
        input1 = Input(shape=(4, 11, 1))
        x1 = Conv2D(16, (2, 5), activation="relu")(input1)
        # x1 = Conv1D(2, 3, activation="relu")(x1)
        x1 = MaxPooling2D((2, 2))(x1)
        x1 = Flatten()(x1)

        input2 = Input(shape=(7, ))
        x2 = Dense(32, activation="relu")(input2)
        x2 = Dense(8, activation="relu")(x2)

        concat = Concatenate()([x1, x2])
        out = Dense(13, activation="softmax")(concat)
        self._target_net = Model(inputs=[input1, input2], outputs=out)
        self._policy_net: Model = tf.keras.models.clone_model(self._target_net)
        self._target_net.compile(loss=self.loss_fn, optimizer="adam")
        self._policy_net.compile(loss=self.loss_fn, optimizer="adam")

    def copy_weights(self):
        self._target_net.set_weights(self._policy_net.get_weights())

    def train(self, replay_memory: typing.Sequence[Memory]):
        states, next_states, next_allowed, actions, rewards, term = sample_replay_memory(replay_memory, model_type=self)
        map_, state = (np.array(x) for x in zip(*states))
        next_map, next_state = (np.array(x) for x in zip(*next_states))

        qsa_prime = np.max(
            self._target_net.predict([next_map, next_state], verbose=0), axis=1, where=next_allowed, initial=-1)
        qsa = self._policy_net.predict([map_, state], verbose=0)
        np.put_along_axis(
            qsa, np.atleast_2d(actions).T, np.atleast_2d(np.where(term, rewards, rewards + 0.99 * qsa_prime)).T, axis=1)
        x = self._policy_net.train_on_batch([map_, state], qsa)
        print(x)

    def get_best_action(self, state: tuple[np.ndarray, np.ndarray], allowed: tuple[bool]):
        map_, state = (np.array(x)[np.newaxis, ...] for x in state)
        pred = self._policy_net.predict([map_, state], verbose=False)
        pred = pred[0]
        valid_idx = np.where(allowed)[0]
        return valid_idx[np.argmax(pred[valid_idx])]


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
    reward: float = None
    terminate: bool = None

    def get_state(self, model: CNNNetworks | Networks):
        """
        big state: 44 bool for each field + error count + 6 dices (2 white (sorted) 4 color) = 51
        small state: 4 distances, 4 crosses per row, error count, dice roll (6) = 15
        cnn state: <<tuple>> 4 x 11 map, error count + 6 dices
        :return:
        """
        if isinstance(model, CNNNetworks):
            return self.map, np.hstack((np.atleast_1d(self.error_count), self.dice_roll))
        if model.input_shape == (None, 15):
            return np.hstack(
                ((dists(self.map) + 1) / 12, n_slc_row(self.map) / 11, np.atleast_1d(self.error_count), self.dice_roll))
        else:
            return np.hstack((self.map.ravel(), np.atleast_1d(self.error_count), self.dice_roll))

    def get_next_state(self, model: CNNNetworks | Networks):
        if isinstance(model, CNNNetworks):
            return self.next_map, np.hstack((np.atleast_1d(self.next_error_count), self.next_dice_roll / 6))
        if model.input_shape == (None, 15):
            return np.hstack(
                ((dists(self.next_map) + 1) / 12, n_slc_row(self.next_map) / 11,
                 np.atleast_1d(self.next_error_count),
                 self.next_dice_roll / 6))
        else:
            return np.hstack((self.next_map.ravel(), np.atleast_1d(self.next_error_count), self.next_dice_roll / 6))


class DQLAgent(_Player):
    games: int = 0

    def __init__(self,
                 name: str,
                 replay_memory: deque[Memory],
                 model: CNNNetworks | Networks):
        super().__init__(name)
        self._score: int = 0
        self._type: typing.Literal["main", "downstream"] = "main"
        self._first_round = True
        self._replay_memory = replay_memory
        self._memory: Memory = Memory(np.empty((1, )), -100, np.empty((1, )))
        self._model: CNNNetworks | Networks = model

    def do_main_move(self, dice_roll: np.ndarray, epsilon: float):
        """
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
            action_idx = self._model.get_best_action(self._memory.get_state(self._model), allowed)
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

    def train_model(self):
        self._model.train(self._replay_memory)

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
        next_allowed += (True,)
        self._memory.next_allowed = next_allowed
        self._memory.next_dice_roll = dice_roll
        self._memory.reward = reward
        self._memory.terminate = terminate

        # print("_____")
        # print(self._memory.state)
        # print(self._memory.action)
        # print(self._memory.next_state)
        self._replay_memory.append(copy.deepcopy(self._memory))

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


if __name__ == "__main__":
    memory = Memory(np.ones((5, )), 4, np.ones((5, )))
    memory.terminate = False
    print(memory)
