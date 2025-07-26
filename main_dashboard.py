import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import subprocess

# Auto-run modules if needed
def run_once_if_missing(path, script):
    if not os.path.exists(path):
        print(f"📦 Running {script} to generate missing file: {path}")
        try:
            subprocess.run(["python", script], check=True)
        except Exception as e:
            print(f"❌ Error running {script}: {e}")

run_once_if_missing("data/predicted_cost_50k.csv", "cost_module.py")
run_once_if_missing("images/dqn_reward_curve.png", "scheduler_module.py")
run_once_if_missing("data/fault_anomalies.csv", "fault_module.py")

# Streamlit page setup
st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Dashboard")
st.caption("Real-time monitoring and analytics for cost prediction, resource scheduling, and fault detection")

# Tabs for each module
tabs = st.tabs(["📊 Cost Prediction", "🧠 Resource Scheduling", "💥 Fault Detection"])

# --- TAB 1: Cost Prediction --- #
with tabs[0]:
    st.subheader("📊 Cost Prediction Module")
    cost_file = "data/predicted_cost_50k.csv"
    metrics_file = "data/cost_metrics.json"

    if os.path.exists(cost_file):
        df_cost = pd.read_csv(cost_file)
        st.dataframe(df_cost.head(10))
        st.success(f"✅ Showing sample of predictions from {cost_file}")
    else:
        st.warning("Cost prediction file not found. Run the module first.")

    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            metrics = json.load(f)

        if 'All Models' in metrics:
            st.markdown("#### 📋 Model Comparison")
            models_df = pd.DataFrame(metrics['All Models']).T
            st.dataframe(models_df.style.format("{:.4f}"))
        else:
            st.metric("MAE", f"{metrics['MAE']:.4f}")
            st.metric("RMSE", f"{metrics['RMSE']:.4f}")
            st.metric("R²", f"{metrics['R2']:.4f}")

        if 'Best Optuna Params' in metrics:
            st.markdown("#### 🔧 Best Optuna Parameters")
            st.json(metrics['Best Optuna Params'])

    if os.path.exists("images/cost_prediction_scatter.png"):
        st.image("images/cost_prediction_scatter.png", caption="Actual vs Predicted Cost")

    if os.path.exists("images/shap_summary.png"):
        st.image("images/shap_summary.png", caption="SHAP Summary Plot")

# --- TAB 2: Resource Scheduling --- #
with tabs[1]:
    st.subheader("🧠 Resource Scheduling Module")

    if os.path.exists("images/dqn_reward_curve.png"):
        st.image("images/dqn_reward_curve.png", caption="DQN Reward Curve")
    else:
        st.warning("Reward curve not found. Run the scheduler module first.")

    st.markdown("#### Benchmark Results")
    if os.path.exists("data/scheduling_metrics.json"):
        with open("data/scheduling_metrics.json") as f:
            sched = json.load(f)
        for key, value in sched.items():
            st.text(f"{key}: {value}")

        st.markdown("#### 📊 Benchmark Comparison Plot")

        models = ["DQN", "FCFS", "Round Robin"]
        wait_times = [
            sched.get("DQN Avg Wait Time", 0),
            sched.get("FCFS Avg Wait Time", 0),
            sched.get("RR Avg Wait Time", 0)
        ]
        rewards = [
            sched.get("DQN Avg Reward", 0),
            sched.get("FCFS Avg Reward", 0),
            sched.get("RR Avg Reward", 0)
        ]

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].barh(models, wait_times, color='skyblue')
        ax[0].set_title("Average Wait Time (s)")
        ax[0].invert_yaxis()

        ax[1].barh(models, rewards, color='lightgreen')
        ax[1].set_title("Average Reward")
        ax[1].invert_yaxis()

        st.pyplot(fig)
    else:
        st.info("Run the scheduling module to generate benchmark metrics.")

# --- TAB 3: Fault Detection --- #
with tabs[2]:
    st.subheader("💥 Fault Detection Module")

    if os.path.exists("data/fault_anomalies.csv"):
        df_fault = pd.read_csv("data/fault_anomalies.csv")
        st.dataframe(df_fault.head(10))
        st.success("✅ Showing anomalies from data/fault_anomalies.csv")
    else:
        st.warning("No fault anomalies CSV found. Run the fault detection module.")

    if os.path.exists("images/fault_detection_plot.png"):
        st.image("images/fault_detection_plot.png", caption="CPU Usage vs Execution Time with Anomalies")

# --- Footer --- #
st.markdown("""
---
👨‍💻 Developed as part of MSc Cloud Computing Thesis Project  
📁 Make sure 'data/' and 'images/' folders are populated by running each module first.
""")
