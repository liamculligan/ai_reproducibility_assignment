import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from statistics import NormalDist

# Create logs directory
if not os.path.exists('logs'):
    os.makedirs('logs')

kappa = 1.5

# Create log files
summary_log_file = os.path.join('logs', f'summary_log_kappa_{kappa}.txt')
detailed_log_file = os.path.join('logs', f'detailed_log_kappa_{kappa}.txt')
ffnn_loss_log_file = os.path.join('logs', f'ffnn_loss_log_kappa_{kappa}.txt')
wunn_loss_log_file = os.path.join('logs', f'wunn_loss_log_kappa_{kappa}.txt')
epistemic_uncertainty_log_file = os.path.join('logs', f'epistemic_uncertainty_log_kappa_{kappa}.txt')
ffnn_prediction_log_file = os.path.join('logs', f'ffnn_prediction_log_kappa_{kappa}.txt')
wunn_prediction_log_file = os.path.join('logs', f'wunn_prediction_log_kappa_{kappa}.txt')
wunn_aleatoric_uncertainty_log_file = os.path.join('logs', f'wunn_aleatoric_uncertainty_log_kappa_{kappa}.txt')

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
        self.num_inputs = 16 * 2 * 4  # 16 tiles, each encoded with 2 4-bit one-hot vectors

    def index_to_coord(self, index):
        dim = int(np.sqrt(16))  # For a 15-puzzle, dim should be 4
        y = index // dim
        x = index % dim
        return x, y

    def encode(self, state):
        dim = int(np.sqrt(len(state)))
        encoded = np.zeros((16, 8))  # 16 tiles, each with 4-bit one-hot vectors for x and y
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
        self.rho_weight = nn.Parameter(torch.Tensor(output_size, input_size).normal_(0, 1))  # TODO: C#
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
    num_iter = 1 #TODO: FIX
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
    learn_heuristic_prac = LearnHeuristicPrac(nnWUNN, nnFFNN, epsilon, max_steps, memory_buffer_max_records, train_iter,
                                              max_train_iter, mini_batch_size, t_max, q, K, learning_rate_ffnn,
                                              learning_rate_wunn, alpha_0)

    # Run the learning algorithm
    trained_ffnn, trained_wunn = learn_heuristic_prac.run(sliding_puzzle, num_iter, num_tasks_per_iter, num_tasks_per_iter_thresh,
                                            delta, beta0, gamma, kappa)

    return trained_ffnn, trained_wunn


trained_ffnn, trained_wunn = run_experiment()

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

    def search(self, start_state, heuristic, t_max=None):
        self.start_time = time.time()

        self.bound = heuristic

        while True:
            self.minoob = -1
            self.path.clear()
            goal = self.dfs(start_state, 0, heuristic,  None, t_max)
            if goal is None:
                return None
            if self.path:
                break
            self.bound = self.minoob

        self.path.reverse()
        elapsed_time = time.time() - self.start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        return self.path, elapsed_time, len(self.path) - 1  # Return time and cost to goal

    def dfs(self, state, cost_so_far, heuristic, pop, t_max):
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
            goal = self.dfs(next_state, cost_so_far + 1, heuristic, -op, t_max)
            self.domain.undo_move(state, op)
            if goal:
                self.path.append(state)
                return True
            if goal is None:
                return None
        return False

def random_walk(state, steps, seed=44):
    if seed is not None:
        random.seed(seed)
    puzzle = SlidingPuzzle(len(state) // 4)
    current_state = state
    for _ in range(steps):
        actions = puzzle.get_possible_moves(current_state)
        action = random.choice(actions)
        current_state = puzzle.apply_move(current_state, action)
    return current_state

# Define the goal state
goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]

# Generate puzzles by taking random steps backwards from the goal state
puzzle_states = [goal_state]
for steps in range(10, 101, 10):
    puzzle_states.append(random_walk(goal_state, steps, seed=44))

for puzzle_state in puzzle_states:

    print(f'puzzle state: {puzzle_state}')

    sliding_puzzle = SlidingPuzzle(size=4)

    puzzle = torch.tensor(sliding_puzzle.encoder.encode(puzzle_state), dtype=torch.float32)

    print(f'puzzle: {puzzle}')

    trained_ffnn.eval()
    with torch.no_grad():
        ffnn_predicted_cost_to_goal = trained_ffnn(puzzle).item()
        print(f'ffn predicted cost to goal: {ffnn_predicted_cost_to_goal}')

    ida_star_test = IDAStarTest(domain=sliding_puzzle,
                            memory_buffer=[],
                            epsilon=1)

    #z_score_test = NormalDist(mu=0, sigma=1).inv_cdf(alpha_test)

    result = ida_star_test.search(puzzle_state, ffnn_predicted_cost_to_goal, t_max=60)

    alphas_test = [0.5]
    for alpha_test in alphas_test:

        #TODO : calc using wunn
        #heuristic = alpha_test * predicted_cost_to_goal

        ida_star_test = IDAStarTest(domain=sliding_puzzle,
                                alpha=alpha_test,
                                memory_buffer=[],
                                epsilon=1)

        #z_score_test = NormalDist(mu=0, sigma=1).inv_cdf(alpha_test)

        result_wunn = ida_star_test.search(puzzle_state, wunn_predicted_cost_to_goal, t_max=60)
