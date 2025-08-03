import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import subprocess

# Auto-run modules
def run_once_if_missing(path, script):
    if not os.path.exists(path):
        print(f"📦 Running {script} to generate: {path}")
        try:
            subprocess.run(["python", script], check=True)
        except Exception as e:
            print(f"❌ Error running {script}: {e}")

run_once_if_missing("data/predicted_cost_50k.csv", "cost_module.py")
run_once_if_missing("images/dqn_reward_curve.png", "scheduler_module.py")
run_once_if_missing("data/fault_anomalies.csv", "fault_module.py")

# UI Setup
st.set_page_config(page_title="Cloud Dashboard", layout="wide")
st.title("☁️ Cloud Intelligence Dashboard")
st.caption("Modules: Cost Prediction, Resource Scheduling, Fault Detection")

tabs = st.tabs(["📊 Cost Prediction", "🧠 Resource Scheduling", "💥 Fault Detection"])

# --- TAB 1: Cost Prediction --- #
with tabs[0]:
    st.subheader("📊 Cost Prediction")
    cost_file = "data/predicted_cost_50k.csv"
    metrics_file = "data/cost_metrics.json"

    if os.path.exists(cost_file):
        df_cost = pd.read_csv(cost_file)
        st.dataframe(df_cost.head(10))
        st.success("✅ Sample from predicted_cost_50k.csv")

    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            metrics = json.load(f)
        if 'All Models' in metrics:
            st.markdown("#### 📋 Model Comparison")
            models_df = pd.DataFrame(metrics['All Models']).T
            st.dataframe(models_df.style.format("{:.4f}"))
        if 'Best Optuna Params' in metrics:
            st.markdown("#### 🔧 Best XGBoost Params")
            st.json(metrics['Best Optuna Params'])

    st.markdown("#### 📸 Visualizations")
    if os.path.exists("images/cost_prediction_scatter.png"):
        st.image("images/cost_prediction_scatter.png", caption="Actual vs Predicted Cost")
    if os.path.exists("images/model_metric_comparison.png"):
            st.image("images/model_metric_comparison.png", caption="Model Metric Comparison")

# --- TAB 2: Resource Scheduling --- #
with tabs[1]:
    st.subheader("🧠 Resource Scheduling")

    if os.path.exists("images/dqn_reward_curve.png"):
        st.image("images/dqn_reward_curve.png", caption="DQN Reward Curve")

    if os.path.exists("data/scheduler_metrics.json"):
        st.markdown("#### 📋 Benchmark Results")
        with open("data/scheduler_metrics.json") as f:
            sched = json.load(f)
        st.code(json.dumps(sched, indent=2))

        st.markdown("#### 📊 Benchmark Visuals")
        
        if os.path.exists("images/reward_comparison_bar.png"):
            st.image("images/reward_comparison_bar.png", caption="Reward Comparison")
        if os.path.exists("images/wait_time_comparison_bar.png"):
            st.image("images/wait_time_comparison_bar.png", caption="Wait Time Comparison")
        if os.path.exists("images/scheduler_radar_chart.png"):
            st.image("images/scheduler_radar_chart.png", caption="Scheduler Radar Chart")

# --- TAB 3: Fault Detection --- #
with tabs[2]:
    st.subheader("💥 Fault Detection")

    if os.path.exists("data/fault_anomalies.csv"):
        df_fault = pd.read_csv("data/fault_anomalies.csv")
        st.dataframe(df_fault.head(10))
        st.success("✅ Showing anomalies from fault_anomalies.csv")

    st.markdown("#### 📸 Fault Analysis Visuals")
    if os.path.exists("images/fault_anomaly_plot.png"):
        st.image("images/fault_anomaly_plot.png", caption="Anomaly Highlighted Distribution")
    if os.path.exists("images/fault_anomaly_boxplot.png"):
        st.image("images/fault_anomaly_boxplot.png", caption="CPU Usage Distribution by Anomaly")
    if os.path.exists("images/fault_anomaly_ratio_pie.png"):
        st.image("images/fault_anomaly_ratio_pie.png", caption="Anomaly vs Normal Ratio")

# --- Footer --- #
st.markdown("""
---
👨‍💻 Developed as part of MSc Cloud Computing Thesis  
📁 Ensure `/data` and `/images` folders are populated by running modules
""")
