import keras.optimizers
from Player import Player
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
class ReplayMemory:
    state: np.ndarray
    next_state: np.ndarray
    action: int
    reward: int
    terminate: bool


class Networks:
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    def __init__(self, input_shape: tuple[int], output: int, hidden_layer_neurons: tuple[int, ...]):
        self._target_net: Sequential = Sequential()
        self._target_net.add(Input(shape=input_shape))
        for hl in hidden_layer_neurons:
            self._target_net.add(Dense(hl, activation="relu"))
        self._target_net.add(Dense(output, activation="linear"))

        self._policy_net: Sequential = tf.keras.models.clone_model(self._target_net)
        self._n_output = output

    def train(self, batch: typing.Sequence[ReplayMemory]):
        states, next_states, actions, rewards, term = zip(
            *((r.state, r.next_state, r.action, r.reward, r.terminate) for r in batch))
        q_s_a_prime = np.max(self._target_net(np.vstack(next_states), training=True), axis=1)
        q_s_a_target = np.where(term, rewards, rewards + 0.99 * q_s_a_prime)
        q_s_a_target = tf.convert_to_tensor(q_s_a_target, dtype="float32")

        with tf.GradientTape() as tape:
            q_s_a = tf.reduce_sum(
                self._policy_net(np.vstack(states)) * tf.one_hot(actions, self._n_output), axis=1)
            loss = Networks.loss_fn(q_s_a_target, q_s_a)
        grads = tape.gradient(loss, self._policy_net.trainable_weights)
        Networks.optimizer.apply_gradients(zip(grads, self._policy_net.trainable_variables))

    def copy_weights(self):
        self._target_net.set_weights(self._policy_net.get_weights())

    def get_best_action(self, state, allowed: tuple[bool]):
        pred = self._policy_net.predict(state)
        pred = pred[0]
        valid_idx = np.where(allowed)[0]
        return valid_idx[np.argmax(pred[valid_idx])]


n_states_main = 50
n_actions_main = 45

main_net = Networks((50, ), 45, (32, 8, 32))
downstream_net = Networks((45, ), 5, (32, 8, 3))


class DQLAgent(Player):
    main_replay: deque[ReplayMemory] = deque([], maxlen=1000)
    downstream_replay: deque[ReplayMemory] = deque([], maxlen=1000)
    games: int = 0

    def __init__(self, name: str):
        super().__init__(name)
        self._score: int = 0
        self._state = None
        self._next_state = None
        self._action = None
        self._type: typing.Literal["main", "downstream"] = "main"

        self._epsilon = 1.0

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
        if random.random() < self._epsilon:
            action_idx = self._select_random_action(allowed)
        else:
            action_idx = main_net.get_best_action(np.atleast_2d(self._state), allowed)
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
        if random.random() < self._epsilon:
            action_idx = self._select_random_action(allowed)
        else:
            action_idx = downstream_net.get_best_action(np.atleast_2d(self._state), allowed)
        self._action = action_idx
        action = moves[action_idx]
        super().take_action(action)
        return action

    def _select_random_action(self, allowed):
        return np.random.choice(np.where(allowed)[0])

    def _add_to_replay(self, reward: int, terminate: bool = False):
        if self._type == "main":
            dice_roll = throw_dice()
            dice_roll[:2].sort()
            dice_roll = dice_roll / 6
            next_state = np.concatenate((self.env.sel_fields.ravel().astype(int),
                                        dice_roll))
            memory = DQLAgent.main_replay
        else:
            dice_roll = np.sum(throw_dice()[:2]) / 12
            next_state = np.append(self.env.sel_fields.ravel().astype(int), dice_roll)
            memory = DQLAgent.downstream_replay
        memory.append(ReplayMemory(state=self._state,
                                   next_state=next_state,
                                   action=self._action,
                                   reward=reward,
                                   terminate=terminate))

    def end_round_callback(self):
        new_score = self.env.compute_total_score()
        reward = new_score - self._score
        self._add_to_replay(reward)
        self._score = new_score

    def end_game_callback(self, end_game_reward: int):
        reward = end_game_reward
        self._add_to_replay(reward, True)  # final state -- what to do?
        main_net.train(random.sample(DQLAgent.main_replay, 10))
        downstream_net.train(random.sample(DQLAgent.downstream_replay, 10))
        print(self.env.compute_total_score())
        self._epsilon -= 0.0001
        self.env.reset()
        if DQLAgent.games % 10:
            main_net.copy_weights()
            downstream_net.copy_weights()
        DQLAgent.games += 1

    @property
    def is_real(self):
        return False
