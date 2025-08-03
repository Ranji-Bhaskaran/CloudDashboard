import time
start_time = time.time()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import json
from math import pi

# STEP 1: Load dataset
df = pd.read_csv("data/cloud_task_scheduling_dataset_20k.csv")

# STEP 2: Define custom Gym environment
class TaskSchedulingEnv(gym.Env):
    def __init__(self, df, num_vms=5):
        super(TaskSchedulingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.num_tasks = len(df)
        self.num_vms = num_vms
        self.current_task = 0
        self.vm_queues = [0.0 for _ in range(num_vms)]
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(num_vms)

    def _get_state(self):
        row = self.df.iloc[self.current_task]
        return np.array([
            row['CPU_Usage (%)'] / 100,
            row['RAM_Usage (MB)'] / 16384,
            row['Disk_IO (MB/s)'] / 100,
            row['Network_IO (MB/s)'] / 100,
            row['Priority'] / 3,
            row['Execution_Time (s)'] / 100
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_task = 0
        self.vm_queues = [0.0 for _ in range(self.num_vms)]
        return self._get_state(), {}

    def step(self, action):
        row = self.df.iloc[self.current_task]
        exec_time = row['Execution_Time (s)']
        priority = row['Priority']
        wait_time = self.vm_queues[action]
        total_time = wait_time + exec_time
        self.vm_queues[action] += exec_time
        reward = - (total_time / 1000) * (1 / (priority + 1))
        self.current_task += 1
        terminated = self.current_task >= self.num_tasks
        truncated = False
        next_state = self._get_state() if not terminated else np.zeros(6, dtype=np.float32)
        info = {"wait_time": wait_time, "total_time": total_time}
        return next_state, reward, terminated, truncated, info

# STEP 3: Train DQN Agent
env = DummyVecEnv([lambda: Monitor(TaskSchedulingEnv(df))])
model = DQN("MlpPolicy", env, learning_rate=0.001, buffer_size=10000, batch_size=64,
            exploration_fraction=0.2, verbose=0)
model.learn(total_timesteps=50000)

# STEP 4: Evaluate Agent
obs = env.reset()
total_reward = 0
reward_log = []
wait_times = []
throughput = 0
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    reward_log.append(reward)
    wait_times.append(info[0].get("wait_time", 0))
    throughput += 1

# STEP 5: Save Reward Curve
os.makedirs("images", exist_ok=True)
plt.figure(figsize=(10, 5))
plt.plot(reward_log, label='DQN Reward per Task', alpha=0.8)
plt.axhline(y=np.mean(reward_log), color='red', linestyle='--', label='Average Reward')
plt.title("📈 DQN Reward per Task Over Time")
plt.xlabel("Task Number")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("images/dqn_reward_curve.png")
plt.close()

# STEP 6: FCFS and RR Baselines
def evaluate_baseline(df, strategy='fcfs'):
    queues = [0.0] * 5
    rewards = []
    for i, row in df.iterrows():
        exec_time = row['Execution_Time (s)']
        priority = row['Priority']
        vm_id = i % 5
        wait_time = queues[vm_id]
        queues[vm_id] += exec_time
        total_time = wait_time + exec_time
        reward = - (total_time / 1000) * (1 / (priority + 1))
        rewards.append(reward)
    return np.mean(rewards), np.mean(queues)

fcfs_avg_reward, fcfs_avg_wait = evaluate_baseline(df, 'fcfs')
rr_avg_reward, rr_avg_wait = evaluate_baseline(df, 'rr')

# STEP 7: Save Metrics
metrics = {
    "Tasks Scheduled": throughput,
    "DQN Cumulative Reward": float(np.sum(reward_log)),
    "DQN Avg Reward": float(np.mean(reward_log)),
    "DQN Avg Wait Time": float(np.mean(wait_times)),
    "FCFS Avg Reward": float(fcfs_avg_reward),
    "FCFS Avg Wait Time": float(fcfs_avg_wait),
    "RR Avg Reward": float(rr_avg_reward),
    "RR Avg Wait Time": float(rr_avg_wait)
}
os.makedirs("data", exist_ok=True)
with open("data/scheduler_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# STEP 8: Additional Visualizations
# Bar Plot - Avg Reward
plt.figure(figsize=(8,5))
models = ["DQN", "FCFS", "Round Robin"]
avg_rewards = [metrics["DQN Avg Reward"], metrics["FCFS Avg Reward"], metrics["RR Avg Reward"]]
plt.bar(models, avg_rewards, color=["skyblue", "orange", "green"])
plt.title("Average Reward Comparison")
plt.ylabel("Average Reward")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("images/reward_comparison_bar.png")
plt.close()

# Bar Plot - Avg Wait Time
plt.figure(figsize=(8,5))
avg_waits = [metrics["DQN Avg Wait Time"], metrics["FCFS Avg Wait Time"], metrics["RR Avg Wait Time"]]
plt.bar(models, avg_waits, color=["skyblue", "orange", "green"])
plt.title("Average Wait Time Comparison")
plt.ylabel("Average Wait Time (s)")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("images/wait_time_comparison_bar.png")
plt.close()

# Radar Chart
categories = ['Avg Reward', 'Avg Wait Time']
values_dqn = [metrics["DQN Avg Reward"], metrics["DQN Avg Wait Time"]]
values_fcfs = [metrics["FCFS Avg Reward"], metrics["FCFS Avg Wait Time"]]
values_rr = [metrics["RR Avg Reward"], metrics["RR Avg Wait Time"]]

labels = np.array(categories)
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]
values_dqn += values_dqn[:1]
values_fcfs += values_fcfs[:1]
values_rr += values_rr[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.plot(angles, values_dqn, label='DQN', color='skyblue', linewidth=2)
ax.fill(angles, values_dqn, alpha=0.25, color='skyblue')
ax.plot(angles, values_fcfs, label='FCFS', color='orange', linewidth=2)
ax.fill(angles, values_fcfs, alpha=0.25, color='orange')
ax.plot(angles, values_rr, label='RR', color='green', linewidth=2)
ax.fill(angles, values_rr, alpha=0.25, color='green')

ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
plt.title("Scheduler Comparison: Reward vs Wait Time")
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig("images/scheduler_radar_chart.png")
plt.close()

# STEP 9: Summary
print("\\n" + "="*60)
print("✅ FINAL SCHEDULING BENCHMARK")
print("="*60)
print(f"📦 Tasks Scheduled         : {throughput}")
print(f"🏆 DQN Cumulative Reward   : {metrics['DQN Cumulative Reward']:.2f}")
print(f"📈 DQN Avg Reward          : {metrics['DQN Avg Reward']:.2f}")
print(f"⏱️  DQN Avg Wait Time       : {metrics['DQN Avg Wait Time']:.2f} s")
print("\\n🆚 FCFS Baseline:")
print(f"📉 FCFS Avg Reward         : {fcfs_avg_reward:.2f}")
print(f"⏱️  FCFS Avg Wait Time      : {fcfs_avg_wait:.2f} s")
print("\\n🔁 Round-Robin Baseline:")
print(f"📉 RR Avg Reward           : {rr_avg_reward:.2f}")
print(f"⏱️  RR Avg Wait Time        : {rr_avg_wait:.2f} s")
print("="*60)
print("📸 Visuals saved in /images")
print("="*60)

# STEP 10: Runtime
end_time = time.time()
print(f"⏱️ Runtime: {end_time - start_time:.2f} seconds")
