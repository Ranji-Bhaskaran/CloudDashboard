# STEP 0: Runtime Logging Start
import time
start_time = time.time()

# STEP 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import json
import gdown

# STEP 1.5: Download vmCloud_data.csv from Google Drive if not present
os.makedirs("data", exist_ok=True)
vmcloud_path = "data/vmCloud_data.csv"
if not os.path.exists(vmcloud_path):
    print("⬇️ Downloading vmCloud_data.csv from Google Drive...")
    gdown.download("https://drive.google.com/uc?id=1_V3uAShSGwjYzxdFm9QURrF-UqxQh9on", vmcloud_path, quiet=False)

# STEP 2: Load Dataset
df = pd.read_csv(vmcloud_path)

# STEP 3: Drop Irrelevant Columns
drop_cols = ['timestamp', 'task_type', 'task_status', 'task_prior']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# STEP 4: Encode VM ID
df['vm_id'] = df['vm_id'].astype(str)
df['vm_id_encoded'] = LabelEncoder().fit_transform(df['vm_id'])

# STEP 5: Feature Engineering
df['cpu_mem_ratio'] = df['cpu_usage'] / (df['memory_usage'] + 1e-5)
df['power_exec_ratio'] = df['power_consumption'] / (df['execution_time'] + 1e-5)
df['network_per_task'] = df['network_traffic'] / (df['num_executed_instructions'] + 1e-5)
df['log_execution'] = np.log1p(df['execution_time'])
df['log_power'] = np.log1p(df['power_consumption'])
df['log_net'] = np.log1p(df['network_traffic'])

# STEP 6: Simulate Time-Series & Temporal Features
df = df.sort_values(by='execution_time')
df['rolling_cpu'] = df['cpu_usage'].rolling(window=3, min_periods=1).mean()
df['cpu_diff'] = df['cpu_usage'].diff().fillna(0)
df['mem_diff'] = df['memory_usage'].diff().fillna(0)

# STEP 7: Select Final Features
features = [
    'vm_id_encoded', 'cpu_usage', 'memory_usage', 'network_traffic',
    'power_consumption', 'num_executed_instructions', 'execution_time',
    'energy_efficiency', 'cpu_mem_ratio', 'power_exec_ratio',
    'network_per_task', 'log_execution', 'log_power', 'log_net',
    'rolling_cpu', 'cpu_diff', 'mem_diff'
]

# STEP 8: Normalize and Train Isolation Forest
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[features])

model = IsolationForest(
    n_estimators=300, max_samples='auto',
    contamination=0.015, random_state=42,
    bootstrap=True, n_jobs=-1
)
df['prediction'] = model.fit_predict(X_scaled)
df['anomaly'] = df['prediction'].map({1: 0, -1: 1})

# STEP 9: Save Results
os.makedirs("images", exist_ok=True)

anomaly_count = int(df['anomaly'].sum())
df[df['anomaly'] == 1].to_csv("data/fault_anomalies.csv", index=False)
print(f"✅ Anomalies detected: {anomaly_count} out of {len(df)} rows")
print("📁 Anomaly results saved to: data/fault_anomalies.csv")

# STEP 10: Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='execution_time', y='cpu_usage', hue='anomaly', palette='Set1', alpha=0.8)
plt.title("💥 Fault Detection: CPU Usage vs Execution Time")
plt.xlabel("Execution Time")
plt.ylabel("CPU Usage")
plt.grid(True)
plt.tight_layout()
plt.savefig("images/fault_detection_plot.png")
plt.close()
print("📸 Fault detection plot saved to: images/fault_detection_plot.png")

# STEP 11: Save Metrics to JSON
metrics = {
    "Total Samples": int(len(df)),
    "Anomalies Detected": anomaly_count,
    "Anomaly Ratio (%)": round(anomaly_count / len(df) * 100, 2)
}
with open("data/fault_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# STEP 12: Runtime Logging End
end_time = time.time()
print(f"⏱️ Runtime: {end_time - start_time:.2f} seconds")
