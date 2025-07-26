import subprocess
import time

print("🔁 Step 1: Running cost_module.py...")
start = time.time()
subprocess.run(["python", "cost_module.py"], check=True)
print(f"✅ cost_module.py completed in {time.time() - start:.2f}s\n")

print("🔁 Step 2: Running scheduler_module.py...")
start = time.time()
subprocess.run(["python", "scheduler_module.py"], check=True)
print(f"✅ scheduler_module.py completed in {time.time() - start:.2f}s\n")

print("🔁 Step 3: Running fault_module.py...")
start = time.time()
subprocess.run(["python", "fault_module.py"], check=True)
print(f"✅ fault_module.py completed in {time.time() - start:.2f}s\n")

print("🚀 Launching Streamlit Dashboard...")
subprocess.run(["streamlit", "run", "main_dashboard.py", "--server.headless", "true"], check=True)
