# model.py

import pandas as pd
import numpy as np

# Load your waypoints from CSV
waypoints_df = pd.read_csv("waypoints_with_scores.csv")

# Assign random eco-friendly scores if not already present
if 'Eco_Friendly_Score' not in waypoints_df.columns:
    np.random.seed(42)  # For reproducibility
    waypoints_df['Terrain_Score'] = np.random.randint(1, 10, len(waypoints_df))
    waypoints_df['AQI_Score'] = np.random.randint(1, 10, len(waypoints_df))
    waypoints_df['Traffic_Congestion_Score'] = np.random.randint(1, 10, len(waypoints_df))
    waypoints_df['Eco_Friendly_Score'] = (
        waypoints_df['Terrain_Score'] +
        waypoints_df['AQI_Score'] +
        waypoints_df['Traffic_Congestion_Score']
    )

# Q-Learning Agent Class
class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))  # Initialize Q-table
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def choose_action(self, state, available_actions):
        if np.random.rand() < self.epsilon:
            return np.random.choice(available_actions)  # Explore
        else:
            state_q_values = self.q_table[state, available_actions]
            return available_actions[np.argmax(state_q_values)]  # Exploit

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

class EcoFriendlyRouteEnv:
    def __init__(self, waypoints):
        if len(waypoints) < 2:
            raise ValueError("At least two waypoints are required.")
        self.waypoints = waypoints
        self.start = 0
        self.end = len(waypoints) - 1
        self.current_state = self.start
        self.done = False

    def reset(self):
        self.current_state = self.start
        self.done = False
        return self.current_state

    def step(self, action):
        next_waypoint = action
        reward = self.get_reward(next_waypoint)
        self.current_state = next_waypoint
        self.done = (next_waypoint == self.end)
        return self.current_state, reward, self.done

    def get_reward(self, waypoint_index):
        return -self.waypoints.iloc[waypoint_index]['Eco_Friendly_Score']

    def available_actions(self, visited):
        return [i for i in range(len(self.waypoints)) if i not in visited]

def get_best_route(env, agent):
    state = env.reset()
    route = []
    visited = set()
    done = False

    while not done:
        route.append(state)
        visited.add(state)
        available_actions = env.available_actions(visited)
        if not available_actions:
            break
        action = agent.choose_action(state, available_actions)
        next_state, _, done = env.step(action)
        state = next_state

    return [env.waypoints.iloc[i][['Latitude', 'Longitude']].tolist() for i in route]

def train(env, agent, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        done = False
        visited = set()

        while not done:
            available_actions = env.available_actions(visited)
            action = agent.choose_action(state, available_actions)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            visited.add(state)
            state = next_state
