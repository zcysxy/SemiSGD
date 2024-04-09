import numpy as np
import torch
from torch import nn
import random
import time


class Actor(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        random.seed(time.time())
        self.model = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(torch.from_numpy(x).float())


def value_iteration_ddpg(rho, u_max):
    iteration = 30
    n_cell = rho.shape[0]
    T_terminal = int(rho.shape[1] / rho.shape[0])
    delta_T = 1 / n_cell
    T = int(T_terminal / delta_T)
    u = np.zeros((n_cell, T))
    V = np.zeros((n_cell + 1, T + 1))

    actor = Actor(2)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)

    # use value iteration to get the expected V table
    for _ in range(iteration):
        for i in range(n_cell):
            for t in range(T):
                u[i, t] = (V[i, t + 1] - V[i + 1, t + 1]) / delta_T + 1 - rho[i, t]
                u[i, t] = min(max(u[i, t], 0), 1)
                V[i, t] = delta_T * (0.5 * u[i, t] ** 2 + rho[i, t] * u[i, t] - u[i, t]) + (1 - u[i, t]) * V[i, t + 1] + \
                          u[i, t] * V[i + 1, t + 1]

            for t in range(T + 1):
                V[(n_cell, t)] = V[(0, t)]

    for _ in range(500):
        states = list()
        a = list()
        b = list()
        c = list()
        i, t = np.random.rand(1)[0], 0. #np.random.choice(range(T))
        while t < T and i < n_cell - 1:
            state = np.array([i, t])
            states.append(state)
            speed = actor.forward(state)
            V_state = (1 - i + int(i)) * V[int(i), int(t)] + (i - int(i)) * V[int(i) + 1, int(t)]
            rho_state = (1 - i + int(i)) * rho[int(i), int(t)] + (i - int(i)) * rho[int(i) + 1, int(t)]
            a.append(delta_T / 2)
            tmp_b = rho_state - 1
            tmp_c = -V_state
            if i + speed >= int(i) + 1:
                tmp_b += V[int(i) + 2, int(t + 1)] - V[int(i) + 1, int(t + 1)]
                tmp_c += (i - int(i) - 1) * V[int(i) + 2, int(t + 1)] + (2 - i + int(i)) * V[int(i) + 1, int(t + 1)]
            else:
                tmp_b += V[int(i) + 1, int(t + 1)] - V[int(i), int(t + 1)]
                tmp_c += (i - int(i)) * V[int(i) + 1, int(t + 1)] + (1 - i + int(i)) * V[int(i), int(t + 1)]

            b.append(tmp_b)
            c.append(tmp_c)
            i += float(speed)
            t += 1.

        states = np.array(states)
        speeds = actor.forward(states)
        a = torch.tensor(np.reshape(np.array(a), (len(states), 1)))
        b = torch.tensor(np.reshape(np.array(b), (len(states), 1)))
        c = torch.tensor(np.reshape(np.array(c), (len(states), 1)))
        advantages = a * speeds ** 2 + b * speeds + c
        policy_loss = advantages.mean()
        actor_optimizer.zero_grad()
        policy_loss.backward()
        actor_optimizer.step()


    u_new = np.zeros((n_cell, T))
    V_new = np.zeros((n_cell + 1, T + 1), dtype=np.float64)
    for i in range(n_cell + 1):
        for t in range(T + 1):
            if i < n_cell and t < T:
                u_new[i, t] = actor(np.array([i, t]))

            V_new[i, t] = V[i, t]

    return u_new, V_new
