# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:03:50 2024

@author: liamc
"""

import os
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from statistics import NormalDist

# Create logs directory
if not os.path.exists('logs'):
    os.makedirs('logs')

kappa = 0.64

# Create log files
summary_log_file = os.path.join('logs', f'summary_log_kappa_{kappa}.txt')
detailed_log_file = os.path.join('logs', f'detailed_log_kappa_{kappa}.txt')
ffnn_loss_log_file = os.path.join('logs', f'ffnn_loss_log_kappa_{kappa}.txt')
wunn_loss_log_file = os.path.join('logs', f'wunn_loss_log_kappa_{kappa}.txt')
epistemic_uncertainty_log_file = os.path.join('logs', f'epistemic_uncertainty_log_kappa_{kappa}.txt')
ffnn_prediction_log_file = os.path.join('logs', f'ffnn_prediction_log_kappa_{kappa}.txt')
wunn_prediction_log_file = os.path.join('logs', f'wunn_prediction_log_kappa_{kappa}.txt')
wunn_aleatoric_uncertainty_log_file = os.path.join('logs', f'wunn_aleatoric_uncertainty_log_kappa_{kappa}.txt')
test_results_log_file = os.path.join('logs', f'test_results_log_kappa_{kappa}.txt')

# Delete log files if they exist
if os.path.exists(summary_log_file):
    os.remove(summary_log_file)
if os.path.exists(detailed_log_file):
    os.remove(detailed_log_file)
if os.path.exists(ffnn_loss_log_file):
    os.remove(ffnn_loss_log_file)
if os.path.exists(wunn_loss_log_file):
    os.remove(wunn_loss_log_file)
if os.path.exists(epistemic_uncertainty_log_file):
    os.remove(epistemic_uncertainty_log_file)
if os.path.exists(ffnn_prediction_log_file):
    os.remove(ffnn_prediction_log_file)
if os.path.exists(wunn_prediction_log_file):
    os.remove(wunn_prediction_log_file)
if os.path.exists(wunn_aleatoric_uncertainty_log_file):
    os.remove(wunn_aleatoric_uncertainty_log_file)
if os.path.exists(test_results_log_file):
    os.remove(test_results_log_file)

def log_to_file(log_file, message):
    with open(log_file, 'a') as f:
        f.write(message + '\n')

def log_to_console_and_file(log_file, message):
    print(message)
    log_to_file(log_file, message)

# Set random seeds for reproducibility
def set_random_seeds(seed=17):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SlidingPuzzle:
    def __init__(self, size=4, ffnn=None, wunn=None):
        self.size = size
        self.goal_state = self.generate_goal_state()
        self.encoder = TwoDim(size)
        self.ffnn = ffnn
        self.wunn = wunn
        self.init = self.goal_state.copy()

    def generate_goal_state(self):
        return list(range(1, self.size * self.size)) + [0]

    def is_goal(self, state):
        return state == self.goal_state

    def get_possible_moves(self, state):
        moves = []
        zero_index = state.index(0)
        row, col = divmod(zero_index, self.size)
        if row > 0: moves.append(-self.size)  # Up
        if row < self.size - 1: moves.append(self.size)  # Down
        if col > 0: moves.append(-1)  # Left
        if col < self.size - 1: moves.append(1)  # Right
        return moves

    def apply_move(self, state, move):
        zero_index = state.index(0)
        new_index = zero_index + move
        new_state = state.copy()
        new_state[zero_index], new_state[new_index] = new_state[new_index], new_state[zero_index]
        return new_state

    def reverse_move(self, state, move):
        zero_index = state.index(0)
        new_index = zero_index - move
        if 0 <= new_index < len(state):
            new_state = state.copy()
            new_state[zero_index], new_state[new_index] = new_state[new_index], new_state[zero_index]
            return new_state
        return state  # Return the original state if the new index is out of bounds

    def undo_move(self, state, move):
        return self.reverse_move(state, -move)

    def evaluate(self, state):
        x = torch.tensor(self.encoder.encode(state), dtype=torch.float32)

        # FFNN output
        self.ffnn.eval()
        ffnn_pred_cost = self.ffnn(x).item()

        log_to_file(ffnn_prediction_log_file, f'Predicted cost: {ffnn_pred_cost}')

        self.wunn.eval()
        wunn_output = self.wunn(x, sample=False)
        log_aleatoric_uncertainty = wunn_output[1]
        aleatoric_uncertainty = torch.log1p(torch.exp(log_aleatoric_uncertainty)).item()

        log_to_file(wunn_aleatoric_uncertainty_log_file, f'Predicted aleatoric uncertainty: {aleatoric_uncertainty}')

        return ffnn_pred_cost, aleatoric_uncertainty

    def get_reverse_effect_set(self, state):
        reverse_set = []
        zero_index = state.index(0)
        row, col = divmod(zero_index, self.size)
        if row > 0:  # Move blank up
            new_state = list(state)
            new_state[zero_index], new_state[zero_index - self.size] = new_state[zero_index - self.size], new_state[
                zero_index]
            reverse_set.append(tuple(new_state))
        if row < self.size - 1:  # Move blank down
            new_state = list(state)
            new_state[zero_index], new_state[zero_index + self.size] = new_state[zero_index + self.size], new_state[
                zero_index]
            reverse_set.append(tuple(new_state))
        if col > 0:  # Move blank left
            new_state = list(state)
            new_state[zero_index], new_state[zero_index - 1] = new_state[zero_index - 1], new_state[zero_index]
            reverse_set.append(tuple(new_state))
        if col < self.size - 1:  # Move blank right
            new_state = list(state)
            new_state[zero_index], new_state[zero_index + 1] = new_state[zero_index + 1], new_state[zero_index]
            reverse_set.append(tuple(new_state))

        # Generate_task_prac gets stuck in a dead end if these are not shuffled, as s0 is always set to final s
        random.shuffle(reverse_set)
        return reverse_set

    def generate_task_prac(self, nnWUNN, epsilon, max_steps, K, added_states):

        # Initialise s0 as goal state
        s0 = self.generate_goal_state()

        s = None
        s_prev = None
        num_steps = 0

        goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]

        while num_steps < max_steps:
            num_steps += 1
            states = {}

            # Get all possible reverse moves
            reverse_moves = self.get_reverse_effect_set(s0)
            for s in reverse_moves:

                log_to_file(detailed_log_file, f'Generated state: {s}')

                if np.array_equal(goal_state, s):
                    continue

                # Don't add identical puzzle configurations to memory buffer
                if list(s) in added_states:
                    log_to_file(detailed_log_file, f'State the same as previously added, continuing')
                    continue

                # Don't include state that takes us back to the previously observed state
                if s_prev is not None and np.array_equal(s_prev, s):
                    log_to_file(detailed_log_file, f'State the same as previous, continuing')
                    continue

                x = torch.tensor(self.encoder.encode(s), dtype=torch.float32).unsqueeze(0)
                epistemic_uncertainty = get_epistemic_uncertainty(x, nnWUNN, K)
                log_to_file(detailed_log_file, f'State epistemic uncertainty: {epistemic_uncertainty}')

                log_to_file(epistemic_uncertainty_log_file, f'State: {s}, epistemic uncertainty: {epistemic_uncertainty}')

                states[tuple(s)] = epistemic_uncertainty

            if len(states) > 0:
                # Sample from softmax distribution derived from states.Values
                state_values = np.array(list(states.values()))
                softmax_probs = np.exp(state_values) / np.sum(np.exp(state_values))
                selected_index = np.random.choice(len(states), p=softmax_probs)
                selected_state = list(list(states.keys())[selected_index])
                selected_uncertainty = state_values[selected_index]

                log_to_file(detailed_log_file, f'Selected State: {selected_state}')
                log_to_file(detailed_log_file, f'Selected epistemic uncertainty: {selected_uncertainty}')

                if selected_uncertainty >= epsilon:
                    T = {'initial': selected_state, 'steps': num_steps}
                    return T

            # Update s_prev
            s_prev = s0
            # Update s0 to current state, if it exists
            if s:
                s0 = list(s)

        # If reached end of max_steps
        return None

class TwoDim:
    def __init__(self, size):
        self.size = size
        # 16 tiles, each encoded with 2 4-bit one-hot vectors
        self.num_inputs = 16 * 2 * 4  

    def index_to_coord(self, index):
        dim = int(np.sqrt(16))
        y = index // dim
        x = index % dim
        return x, y

    def encode(self, state):
        dim = int(np.sqrt(len(state)))
        encoded = np.zeros((16, 8))
        for i, val in enumerate(state):
            if val == 0:
                continue
            x, y = self.index_to_coord(i)
            encoded[val - 1, x] = 1
            encoded[val - 1, 4 + y] = 1
        return encoded.flatten()

class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.025):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x

    def train_model(self, X, y, epochs, learning_rate):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            outputs = self.forward(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 100 == 0:
                log_to_file(detailed_log_file, f'FFNN epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

            log_to_file(ffnn_loss_log_file, f'Epoch [{epoch + 1}/{epochs}]')
            log_to_file(ffnn_loss_log_file, f'Loss: {loss.item():.4f}')


class WUNNLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(WUNNLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.mu_weight = nn.Parameter(torch.Tensor(output_size, input_size).normal_(0, 1))  # C#
        self.rho_weight = nn.Parameter(torch.Tensor(output_size, input_size).normal_(0, 1))  # C#
        self.mu_bias = nn.Parameter(torch.Tensor(output_size).normal_(0, 1))  # C#
        self.rho_bias = nn.Parameter(torch.Tensor(output_size).normal_(0, 1))  # C#

    def forward(self, x, sample=True):
        if self.training or sample:
            epsilon_weight = torch.randn_like(self.rho_weight)
            epsilon_bias = torch.randn_like(self.rho_bias)
            weight = self.mu_weight + torch.log1p(torch.exp(self.rho_weight)) * epsilon_weight
            bias = self.mu_bias + torch.log1p(torch.exp(self.rho_bias)) * epsilon_bias
        else:
            weight = self.mu_weight
            bias = self.mu_bias
        return F.linear(x, weight, bias)


class WUNN(nn.Module):
    def __init__(self, input_size):
        super(WUNN, self).__init__()
        self.fc1 = WUNNLayer(input_size, 20)
        # Output cost-to-goal and uncertainty
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x, sample=True):
        x = torch.relu(self.fc1(x, sample))
        x = self.fc2(x)
        x = torch.relu(x)
        return x

    def train_model(self, X, y, learning_rate, beta):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.train()
        optimizer.zero_grad()
        loss = self.custom_loss(X, y, beta, 5)
        log_to_file(wunn_loss_log_file, f'Loss: {loss.item():.4f}')
        loss.backward()
        optimizer.step()

    def custom_loss(self, X, y, beta, S):
        # KL Divergence for each layer
        kl = self.kl_divergence(self.fc1.mu_weight, self.fc1.rho_weight) + self.kl_divergence(self.fc1.mu_bias, self.fc1.rho_bias)

        # Monte Carlo estimation of expected log-likelihood
        log_likelihood = 0
        for _ in range(S):
            output = self.forward(X, sample=True)

            mean = output[:, 0]
            log_std = output[:, 1]
            std = torch.log1p(torch.exp(log_std))

            dist = torch.distributions.Normal(mean, std)
            log_likelihood += torch.sum(dist.log_prob(y))

        log_likelihood /= S

        # Combined loss
        loss = beta * kl - log_likelihood
        return loss

    def kl_divergence(self, mu, rho, mu_prior=0, sigma_prior=10**0.5):
        sigma = torch.log1p(torch.exp(rho))
        kl_div = torch.sum(
            torch.log(sigma_prior / sigma) +
            (sigma ** 2 + (mu - mu_prior) ** 2) / (2 * sigma_prior ** 2) -
            0.5
        )
        return kl_div


class IDAStar:
    def __init__(self, domain, alpha, memory_buffer, epsilon):
        self.domain = domain
        self.alpha = alpha
        self.memory_buffer = memory_buffer
        self.epsilon = epsilon
        self.path = []
        self.bound = 0
        self.minoob = -1
        self.start_time = None
        self.expanded = 0
        self.generated = 0

    def search(self, start_state, z_score, t_max=None):
        self.start_time = time.time()

        self.bound, _ = self.domain.evaluate(start_state)

        while True:
            self.minoob = -1
            self.path.clear()
            goal = self.dfs(start_state, 0, None, t_max, z_score)
            if goal is None:
                return None
            if self.path:
                break
            self.bound = self.minoob

        self.path.reverse()
        elapsed_time = time.time() - self.start_time
        log_to_console_and_file(detailed_log_file, f'Search elapsed time: {elapsed_time:.2f} seconds')
        return self.path

    def dfs(self, state, cost_so_far, pop, t_max, z_score):
        if t_max and (time.time() - self.start_time) > t_max:
            return None

        ffnn_pred_cost, alleatoric_uncertainty = self.domain.evaluate(state)

        if len(self.memory_buffer) > 0:
            states, costs = zip(*self.memory_buffer)
            percentile_rank = np.percentile(costs, self.alpha*100)
        else:
            percentile_rank = None

        if percentile_rank and ffnn_pred_cost < percentile_rank:
            heuristic = ffnn_pred_cost - z_score*alleatoric_uncertainty
        else:
            heuristic = ffnn_pred_cost - z_score*self.epsilon

        f = cost_so_far + heuristic

        if f <= self.bound and self.domain.is_goal(state):
            self.path.append(state)
            return True
        if f > self.bound:
            if self.minoob < 0 or f < self.minoob:
                self.minoob = f
            return False

        self.expanded += 1
        for op in self.domain.get_possible_moves(state):
            if op == pop:
                continue
            self.generated += 1
            next_state = self.domain.apply_move(state, op)
            goal = self.dfs(next_state, cost_so_far + 1, -op, t_max, z_score)
            self.domain.undo_move(state, op)
            if goal:
                self.path.append(state)
                return True
            if goal is None:
                return None
        return False


class LearnHeuristicPrac:
    def __init__(self, nnWUNN, nnFFNN, epsilon, max_steps, memoryBufferMaxRecords, trainIter, maxTrainIter,
                 miniBatchSize, t_max, q, K, learning_rate_ffnn, learning_rate_wunn, alpha0):
        self.nnWUNN = nnWUNN
        self.nnFFNN = nnFFNN
        self.memory_buffer = []
        self.memoryBufferMaxRecords = memoryBufferMaxRecords
        self.trainIter = trainIter
        self.maxTrainIter = maxTrainIter
        self.miniBatchSize = miniBatchSize
        self.t_max = t_max
        self.q = q
        self.K = K
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.learning_rate_ffnn = learning_rate_ffnn
        self.learning_rate_wunn = learning_rate_wunn
        self.alpha = alpha0
        self.z_score = NormalDist(mu=0, sigma=1).inv_cdf(self.alpha)
        self.added_states = []

    def run(self, sliding_puzzle, numIter, numTasksPerIter, numTasksPerIterThresh, delta, beta0, gamma, kappa):

        self.beta = beta0
        finished_training = False

        # Training can run for a maximum of 6 hours otherwise stop at end of iteration
        start_time = time.time()
        max_training_time = 6 * 3600

        for n in range(numIter):

            if not finished_training:

                numSolved = 0

                log_to_console_and_file(summary_log_file, f'Iteration {n + 1}/{numIter}')
                log_to_file(detailed_log_file, f'Iteration {n + 1}/{numIter}')
                log_to_file(ffnn_loss_log_file, f'Iteration {n + 1}/{numIter}')
                log_to_file(wunn_loss_log_file, f'Iteration {n + 1}/{numIter}')
                log_to_file(epistemic_uncertainty_log_file, f'Iteration {n + 1}/{numIter}')
                log_to_file(ffnn_prediction_log_file, f'Iteration {n + 1}/{numIter}')
                log_to_file(wunn_prediction_log_file, f'Iteration {n + 1}/{numIter}')
                log_to_file(wunn_aleatoric_uncertainty_log_file, f'Iteration {n + 1}/{numIter}')

                for i in range(numTasksPerIter):
                    T = sliding_puzzle.generate_task_prac(self.nnWUNN, self.epsilon, self.max_steps, self.K, self.added_states)

                    if not T:
                        finished_training = True
                        log_to_file(detailed_log_file, 'Finished training')
                        break

                    log_to_file(detailed_log_file, f'Generated task with initial state {T["initial"]}')

                    solved, plan = self.solve_task(sliding_puzzle, T)

                    if solved:
                        log_to_file(detailed_log_file, f'Solved task with initial state: {T["initial"]}')
                        numSolved += 1

                        log_to_file(detailed_log_file, f'Solved plan: {plan}')

                        plan_length = len(plan)

                        for i, state in enumerate(plan):
                            if not sliding_puzzle.is_goal(state):
                                xj = sliding_puzzle.encoder.encode(state)
                                yj = plan_length-i-1  # Actual cost to goal
                                self.memory_buffer.append((xj, yj))
                                self.added_states.append(state)

                                log_to_file(detailed_log_file, f'Memory buffer size: {len(self.memory_buffer)}')
                                log_to_file(detailed_log_file, f'State: {state}, Cost-to-goal: {yj}')
                    else:
                        log_to_file(detailed_log_file, f'Failed to solve task with initial state: {T["initial"]}')

                if not finished_training:

                    self.memory_buffer = self.memory_buffer[-self.memoryBufferMaxRecords:]

                    # Update alpha and z-score
                    if numSolved < numTasksPerIterThresh:
                        self.alpha = max(self.alpha - delta, 0.5)
                        self.z_score = NormalDist(mu=0, sigma=1).inv_cdf(self.alpha)

                        log_to_console_and_file(summary_log_file, f'NumSolved < NumTasksPerIterThresh, updating alpha to {self.alpha}')
                        log_to_console_and_file(summary_log_file, f'Heuristic z-score {self.z_score} for alpha {self.alpha}')

                        updateBeta = False

                    else:
                        updateBeta = True

                    self.train_nnFFNN()
                    self.train_nnWUNN(updateBeta, kappa, gamma)

                # Check elapsed time
                elapsed_time = time.time() - start_time
                if elapsed_time >= max_training_time:
                    log_to_console_and_file(summary_log_file, f'Maximum training time of 8 hours reached.')
                    finished_training = True
                    break

        return self.nnFFNN, self.nnWUNN

    def solve_task(self, sliding_puzzle, T):
        ida_star = IDAStar(sliding_puzzle, self.alpha, self.memory_buffer, self.epsilon)
        plan = ida_star.search(T['initial'], self.z_score, self.t_max)
        if plan and sliding_puzzle.is_goal(plan[-1]):
            return True, plan
        return False, []

    def train_nnFFNN(self):
        log_to_file(detailed_log_file, f'Training FFNN')
        states, costs = zip(*self.memory_buffer)
        X = torch.tensor(states, dtype=torch.float32)
        y = torch.tensor(costs, dtype=torch.float32).unsqueeze(1)

        self.nnFFNN.train_model(X, y, self.trainIter, self.learning_rate_ffnn)

        log_to_file(detailed_log_file, f'Finished training FFNN')

    def train_nnWUNN(self, updateBeta, kappa, gamma):
        log_to_file(detailed_log_file, f'Training WUNN')
        states, costs = zip(*self.memory_buffer)
        X = torch.tensor(states, dtype=torch.float32)
        y = torch.tensor(costs, dtype=torch.float32).unsqueeze(1)

        early_stop = False

        for iter in range(self.maxTrainIter):

            log_to_file(wunn_loss_log_file, f'Epoch [{iter + 1}/{self.maxTrainIter}]')

            # Train on maximum miniBatchSize observations as per paper
            batch_indices = generate_mini_batch_indices(X, self.miniBatchSize)

            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            self.nnWUNN.train_model(X_batch, y_batch, self.learning_rate_wunn, self.beta)

            state_uncertainties = [get_epistemic_uncertainty(x.unsqueeze(0), self.nnWUNN, self.K) for x in X]

            if all(uncertainty < kappa * self.epsilon for uncertainty in state_uncertainties):
                early_stop = True
                log_to_console_and_file(summary_log_file, f'Early stopping at iteration {iter + 1}')
                break
            if (iter + 1) % 10 == 0:
                log_to_file(detailed_log_file, f'WUNN epoch [{iter + 1}/{self.maxTrainIter}]')

        # Update beta
        if updateBeta and not early_stop:
            self.beta = self.beta*gamma
            log_to_console_and_file(summary_log_file, f'Updating beta to {self.beta}')

        log_to_file(detailed_log_file, f'Finished training WUNN')

def generate_mini_batch_indices(input_list, num_indices):
    list_length = len(input_list)
    if list_length <= num_indices:
        indices = list(range(list_length))
        random.shuffle(indices)
        return indices
    else:
        return random.sample(range(list_length), num_indices)

def get_epistemic_uncertainty(x, model, K=100):
    predictions = []
    model.eval()
    with torch.no_grad():
        for _ in range(K):
            wunn_output = model(x, sample=True)
            wunn_pred_cost = wunn_output[:, 0:1]
            predictions.append(wunn_pred_cost)

            log_to_file(wunn_prediction_log_file, f'Predicted cost: {wunn_pred_cost.item()}')

    predictions = torch.stack(predictions)
    prediction_mean = predictions.mean(0)
    prediction_squared = predictions ** 2
    mean_of_squares = prediction_squared.mean(0)
    epistemic_uncertainty = mean_of_squares - prediction_mean ** 2
    return epistemic_uncertainty.item()

# Run the experiment
def run_experiment():
    epsilon = 1
    num_tasks_per_iter = 10
    num_tasks_per_iter_thresh = 6
    alpha_0 = 0.99
    beta0 = 0.05
    memory_buffer_max_records = 25000
    train_iter = 1000
    max_train_iter = 5000
    mini_batch_size = 100
    t_max = 60
    q = 0.95
    K = 100
    num_iter = 50
    max_steps = 1000
    delta = 0.05
    gamma = (0.00001/beta0)**(1/num_iter)

    # Initialise networks
    set_random_seeds()

    input_size = 16 * 2 * 4  # 15-puzzle with 16 tiles, each encoded with 4 + 4 bits
    hidden_size = 20
    dropout_rate_ffnn = 0.025

    learning_rate_ffnn = 0.001
    learning_rate_wunn = 0.01

    nnFFNN = FFNN(input_size, hidden_size, dropout_rate_ffnn)
    nnWUNN = WUNN(input_size)

    # Initialize SlidingPuzzle
    sliding_puzzle = SlidingPuzzle(size=4, ffnn=nnFFNN, wunn=nnWUNN)

    # Initialise LearnHeuristicPrac
    learn_heuristic_prac = LearnHeuristicPrac(nnWUNN,
                                              nnFFNN,
                                              epsilon,
                                              max_steps,
                                              memory_buffer_max_records,
                                              train_iter,
                                              max_train_iter,
                                              mini_batch_size,
                                              t_max,
                                              q,
                                              K,
                                              learning_rate_ffnn,
                                              learning_rate_wunn,
                                              alpha_0)

    # Run the learning algorithm
    trained_ffnn, trained_wunn = learn_heuristic_prac.run(sliding_puzzle,
                                                          num_iter,
                                                          num_tasks_per_iter,
                                                          num_tasks_per_iter_thresh,
                                                          delta,
                                                          beta0,
                                                          gamma,
                                                          kappa)

    return trained_ffnn, trained_wunn

# Testing models to replicate table 1 in paper
class IDAStarTest:
    def __init__(self, domain, memory_buffer, epsilon):
        self.domain = domain
        self.memory_buffer = memory_buffer
        self.epsilon = epsilon
        self.path = []
        self.bound = 0
        self.minoob = -1
        self.start_time = None
        self.expanded = 0
        self.generated = 0

    def search(self, start_state, heuristic_initial, heuristic_func, t_max=None):
        self.start_time = time.time()
        self.bound = heuristic_initial

        while True:
            self.minoob = -1
            self.path.clear()
            goal = self.dfs(start_state, 0, heuristic_initial, heuristic_func, None, t_max)
            if goal is None:
                if t_max and (time.time() - self.start_time) > t_max:
                    # Return None if no solution found
                    return self.path, time.time() - self.start_time, None
                return None
            if self.path:
                break
            self.bound = self.minoob

        self.path.reverse()
        elapsed_time = time.time() - self.start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        # Return time and cost to goal
        return self.path, elapsed_time, len(self.path) - 1

    def dfs(self, state, cost_so_far, heuristic, heuristic_func, pop, t_max):
        if t_max and (time.time() - self.start_time) > t_max:
            return None

        f = cost_so_far + heuristic

        if f <= self.bound and self.domain.is_goal(state):
            self.path.append(state)
            return True
        if f > self.bound:
            if self.minoob < 0 or f < self.minoob:
                self.minoob = f
            return False

        self.expanded += 1
        for op in self.domain.get_possible_moves(state):
            if op == pop:
                continue
            self.generated += 1
            next_state = self.domain.apply_move(state, op)
            heuristic_next = heuristic_func(next_state)
            goal = self.dfs(next_state, cost_so_far + 1, heuristic_next, 
                            heuristic_func, -op, t_max)
            self.domain.undo_move(state, op)
            if goal:
                self.path.append(state)
                return True
            if goal is None:
                return None
        return False

def generate_puzzles(num_puzzles, initial_state, max_steps=100, seed=44):
    random.seed(seed)
    puzzle = SlidingPuzzle(size=4)
    puzzles = []

    for _ in range(num_puzzles):
        state = initial_state
        for _ in range(random.randint(1, max_steps)):
            moves = puzzle.get_possible_moves(state)
            state = puzzle.apply_move(state, random.choice(moves))
        puzzles.append(state)

    return puzzles

# Check if the solution is optimal
def is_optimal(path, goal_state):
    return int(path[-1] == goal_state) if path else 0

def manhattan_distance(state, goal_state):
    distance = 0
    size = int(len(state) ** 0.5)
    for num in range(1, len(state)):
        current_index = state.index(num)
        goal_index = goal_state.index(num)
        current_row, current_col = divmod(current_index, size)
        goal_row, goal_col = divmod(goal_index, size)
        distance += abs(current_row - goal_row) + abs(current_col - goal_col)
    return distance

def run_tests(trained_ffnn, trained_wunn, alphas_test, optimal_cost=53.05):

    goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]

    # Generate test puzzles
    num_test_puzzles = 10 # Cannot generate 100 within a reasonable amount 
    #of time due to issues with learned heuristics
    test_puzzles = generate_puzzles(num_test_puzzles, goal_state)

    results = []

    for puzzle_state in test_puzzles:
        log_to_console_and_file(test_results_log_file, 
                                f'Puzzle state: {puzzle_state}')

        sliding_puzzle = SlidingPuzzle(size=4, 
                                       ffnn=trained_ffnn, 
                                       wunn=trained_wunn)
        
        puzzle = torch.tensor(sliding_puzzle.encoder.encode(puzzle_state), 
                              dtype=torch.float32)

        # Evaluate FFNN heuristic
        trained_ffnn.eval()
        with torch.no_grad():
            ffnn_predicted_cost_to_goal = trained_ffnn(puzzle).item()
            log_to_console_and_file(test_results_log_file, 
                                    f'FFNN Predicted cost: {ffnn_predicted_cost_to_goal}')

        # Run IDA* for FFNN
        heuristic_func_ffnn = lambda state: trained_ffnn(torch.tensor(sliding_puzzle.encoder.encode(state), 
                                                                      dtype=torch.float32)).item()
        
        ida_star_test_ffnn = IDAStarTest(domain=sliding_puzzle, 
                                         memory_buffer=[], 
                                         epsilon=1)
        
        path_ffnn, time_ffnn, cost_ffnn = ida_star_test_ffnn.search(puzzle_state, 
                                                                    ffnn_predicted_cost_to_goal, 
                                                                    heuristic_func_ffnn, 
                                                                    t_max=60*5)

        elapsed_time_ffnn = time_ffnn if time_ffnn is not None else 0

        if cost_ffnn is None:
            suboptimality_ffnn = None
            optimal_ffnn = 0
        else:
            # Adjust cost to avoid division by zero
            if cost_ffnn == 0:
                cost_ffnn += 1e-6

            # Calculate suboptimality for FFNN
            suboptimality_ffnn = (cost_ffnn / optimal_cost) - 1
            optimal_ffnn = is_optimal(path_ffnn, goal_state)

        log_to_console_and_file(test_results_log_file,
                    f'FFNN Results - Cost: {cost_ffnn}, '
                    f'Time: {elapsed_time_ffnn}, Nodes: {ida_star_test_ffnn.generated}, '
                    f'Path: {path_ffnn}')

        results.append({
            'alpha': 'N/A',  # Single output FFNN has no alpha
            'time': elapsed_time_ffnn,
            'generated': ida_star_test_ffnn.generated,
            'suboptimality': suboptimality_ffnn,
            'optimal': optimal_ffnn
        })

        # Calculate Manhattan Distance heuristic
        manhattan_cost = manhattan_distance(puzzle_state, goal_state)

        heuristic_func_md = lambda state: manhattan_distance(state, goal_state)
        
        ida_star_test_md = IDAStarTest(domain=sliding_puzzle, 
                                       memory_buffer=[], 
                                       epsilon=1)
        
        path_md, time_md, cost_md = ida_star_test_md.search(puzzle_state,
                                                            manhattan_cost, 
                                                            heuristic_func_md, 
                                                            t_max=60*5)

        elapsed_time_md = time_md if time_md is not None else 0

        if cost_md is None:
            suboptimality_md = None
            optimal_md = 0
        else:
            # Calculate suboptimality for Manhattan Distance
            suboptimality_md = (cost_md / optimal_cost) - 1
            optimal_md = is_optimal(path_md, goal_state)

        log_to_console_and_file(test_results_log_file,
                    f'Manhattan Distance Results - Cost: {cost_md}, '
                    f'Time: {elapsed_time_md}, Nodes: {ida_star_test_md.generated}, '
                    f'Path: {path_md}')

        results.append({
            'alpha': 'MD',
            'time': elapsed_time_md,
            'generated': ida_star_test_md.generated,
            'suboptimality': suboptimality_md,
            'optimal': optimal_md
        })

        # Run IDA* for WUNN with different alpha values
        for alpha_test in alphas_test:
            z_score_test = NormalDist(mu=0, sigma=1).inv_cdf(alpha_test)

            trained_wunn.eval()
            with torch.no_grad():
                wunn_output = trained_wunn(puzzle, sample=False)
                wunn_predicted_cost_to_goal = wunn_output[0].item()
                log_aleatoric_uncertainty = wunn_output[1].item()
                aleatoric_uncertainty = torch.log1p(torch.exp(torch.tensor(log_aleatoric_uncertainty))).item()

            heuristic_wunn = wunn_predicted_cost_to_goal - z_score_test * aleatoric_uncertainty
            
            heuristic_func_wunn = lambda state: trained_wunn(torch.tensor(sliding_puzzle.encoder.encode(state), 
                                                                          dtype=torch.float32), sample=False)[0].item() - z_score_test * aleatoric_uncertainty

            log_to_console_and_file(test_results_log_file,
                        f'WUNN Predicted cost: {wunn_predicted_cost_to_goal}, '
                        f'Aleatoric Uncertainty: {aleatoric_uncertainty}, '
                        f'Heuristic: {heuristic_wunn}')

            ida_star_test_wunn = IDAStarTest(domain=sliding_puzzle, 
                                             memory_buffer=[], 
                                             epsilon=1)
            
            path_wunn, time_wunn, cost_wunn = ida_star_test_wunn.search(puzzle_state, 
                                                                        heuristic_wunn, 
                                                                        heuristic_func_wunn, 
                                                                        t_max=60*5)

            elapsed_time_wunn = time_wunn if time_wunn is not None else 0

            if cost_wunn is None:
                suboptimality = None
                optimal_wunn = 0
            else:
                # Calculate suboptimality for WUNN
                suboptimality = (cost_wunn / optimal_cost) - 1
                optimal_wunn = is_optimal(path_wunn, goal_state)

            log_to_console_and_file(test_results_log_file,
                        f'WUNN Results - Cost: {cost_wunn}, '
                        f'Time: {elapsed_time_wunn}, Nodes: {ida_star_test_wunn.generated}, '
                        f'Path: {path_wunn}')

            results.append({
                'alpha': alpha_test,
                'time': elapsed_time_wunn,
                'generated': ida_star_test_wunn.generated,
                'suboptimality': suboptimality,
                'optimal': optimal_wunn
            })

    return results

def average_results(trained_ffnn, trained_wunn, alphas_test, repeats=10):
    all_results = []

    for _ in range(repeats):
        results = run_tests(trained_ffnn, trained_wunn, alphas_test)
        all_results.append(results)

    log_to_console_and_file(test_results_log_file,
                            f'RESULTS: {all_results}')

    # Convert the results to a DataFrame and calculate the mean
    df_all_results = pd.DataFrame([item for sublist in all_results for item in sublist])
    avg_results = df_all_results.groupby(['alpha']).mean().reset_index()

    return avg_results

def plot_training_logs(kappa):

    ffnn_log_path = f'logs/ffnn_loss_log_kappa_{kappa}.txt'
    wunn_log_path = f'logs/wunn_loss_log_kappa_{kappa}.txt'
    epistemic_path = f'logs/epistemic_uncertainty_log_kappa_{kappa}.txt'
    aleatoric_path = f'logs/wunn_aleatoric_uncertainty_log_kappa_{kappa}.txt'
    predicted_cost_path = f'logs/ffnn_prediction_log_kappa_{kappa}.txt'
    kappa = 1.5

    ffnn_data = pd.read_csv(ffnn_log_path, delimiter=':', header=None, names=['Metric', 'Value'])
    wunn_data = pd.read_csv(wunn_log_path, delimiter=':', header=None, names=['Metric', 'Value'])

    # Extract iteration data
    ffnn_iterations = ffnn_data[ffnn_data['Metric'].str.contains('Iteration')].index.tolist()
    wunn_iterations = wunn_data[wunn_data['Metric'].str.contains('Iteration')].index.tolist()

    # Plot FFNN Loss per iteration
    for i in range(len(ffnn_iterations)):
        start_idx = ffnn_iterations[i]
        end_idx = ffnn_iterations[i + 1] if i + 1 < len(ffnn_iterations) else len(ffnn_data)
        epoch_data = ffnn_data.iloc[start_idx + 1:end_idx]

        epoch_data = epoch_data[epoch_data['Metric'].str.contains('Loss')].reset_index(drop=True)
        epoch_data['Epoch'] = epoch_data.index + 1

        plt.figure()
        sns.lineplot(data=epoch_data, x='Epoch', y='Value')
        plt.title(f'FFNN Training Loss Iteration {i + 1}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f'plots/ffnn_loss_iteration_{i + 1}_kappa_{kappa}.png')
        plt.close()

    # Plot WUNN Loss per iteration
    for i in range(len(wunn_iterations)):
        start_idx = wunn_iterations[i]
        end_idx = wunn_iterations[i + 1] if i + 1 < len(wunn_iterations) else len(wunn_data)
        epoch_data = wunn_data.iloc[start_idx + 1:end_idx]

        epoch_data = epoch_data[epoch_data['Metric'].str.contains('Loss')].reset_index(drop=True)
        epoch_data['Epoch'] = epoch_data.index + 1

        plt.figure()
        sns.lineplot(data=epoch_data, x='Epoch', y='Value')
        plt.title(f'WUNN Training Loss Iteration {i + 1}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f'plots/wunn_loss_iteration_{i + 1}_kappa_{kappa}.png')
        plt.close()

    with open(epistemic_path, 'r') as file:
        epistemic_lines = file.readlines()

    epistemic_data = []
    for line in epistemic_lines:
        if 'Iteration' in line:
            iteration = int(line.strip().split()[1].split('/')[0])
        elif 'epistemic uncertainty' in line:
            uncertainty = float(line.strip().split(': ')[-1])
            epistemic_data.append({'Iteration': iteration, 'Epistemic Uncertainty': uncertainty})

    epistemic_df = pd.DataFrame(epistemic_data)

    with open(aleatoric_path, 'r') as file:
        aleatoric_lines = file.readlines()

    aleatoric_data = []
    for line in aleatoric_lines:
        if 'Iteration' in line:
            iteration = int(line.strip().split()[1].split('/')[0])
        elif 'aleatoric uncertainty' in line:
            uncertainty = float(line.strip().split(': ')[-1])
            aleatoric_data.append({'Iteration': iteration, 'Aleatoric Uncertainty': uncertainty})

    aleatoric_df = pd.DataFrame(aleatoric_data)

    with open(predicted_cost_path, 'r') as file:
        predicted_cost_lines = file.readlines()

    predicted_cost_data = []
    for line in predicted_cost_lines:
        if 'Iteration' in line:
            iteration = int(line.strip().split()[1].split('/')[0])
        elif 'Predicted cost' in line:
            cost = float(line.strip().split(': ')[-1])
            predicted_cost_data.append({'Iteration': iteration, 'Predicted Cost': cost})

    predicted_cost_df = pd.DataFrame(predicted_cost_data)

    # Calculate medians
    epistemic_median = epistemic_df.groupby('Iteration')['Epistemic Uncertainty'].median().reset_index()
    aleatoric_median = aleatoric_df.groupby('Iteration')['Aleatoric Uncertainty'].median().reset_index()
    predicted_cost_median = predicted_cost_df.groupby('Iteration')['Predicted Cost'].median().reset_index()

    # Plotting epistemic uncertainty
    plt.figure()
    sns.lineplot(data=epistemic_median, x='Iteration', y='Epistemic Uncertainty', marker='o')
    plt.title('Median Epistemic Uncertainty by Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Median Epistemic Uncertainty')
    plt.savefig(f'plots/median_epistemic_uncertainty_by_iteration_kappa_{kappa}.png')
    plt.show()

    # Plotting aleatoric uncertainty
    plt.figure()
    sns.lineplot(data=aleatoric_median, x='Iteration', y='Aleatoric Uncertainty', marker='o')
    plt.title('Median Aleatoric Uncertainty by Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Median Aleatoric Uncertainty')
    plt.savefig(f'plots/median_aleatoric_uncertainty_by_iteration_kappa_{kappa}.png')
    plt.show()

    # Plotting predicted cost
    plt.figure()
    sns.lineplot(data=predicted_cost_median, x='Iteration', y='Predicted Cost', marker='o')
    plt.title('Median Predicted Cost by Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Median Predicted Cost')
    plt.savefig(f'plots/median_predicted_cost_by_iteration_kappa_{kappa}.png')
    plt.show()

if __name__ == '__main__':

    trained_ffnn, trained_wunn = run_experiment()

    average_results = average_results(trained_ffnn,
                                      trained_wunn,
                                      alphas_test=[0.95,
                                                   0.9, 
                                                   0.75, 
                                                   0.5, 
                                                   0.25, 
                                                   0.1,
                                                   0.05], 
                                      repeats=3)  # Cannot repeat 10 tasks 
    #within a reasonable amount of time due to issues with learned heuristics

    print(average_results)

    average_results.to_csv(f'paper_table_1_results_kappa_{kappa}.csv',
                           index=False)

    plot_training_logs(kappa)