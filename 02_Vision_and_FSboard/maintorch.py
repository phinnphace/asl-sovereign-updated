import subprocess
import torch

def log_hardware_rigor():
    if torch.cuda.is_available():
        # Get the UUID directly from the system via nvidia-smi
        uuid = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], 
            encoding='utf-8'
        ).strip()
        
        device_name = torch.cuda.get_device_name(0)
        
        print(f"[RIGOR] Training on: {device_name}")
        print(f"[RIGOR] Hardware UUID: {uuid}")
        
        # Rigorous move: Save this to a small text file in your output directory
        with open("training_provenance.txt", "w") as f:
            f.write(f"GPU_NAME: {device_name}\n")
            f.write(f"GPU_UUID: {uuid}\n")
            f.write(f"TORCH_VERSION: {torch.__version__}\n")

# Call this at the start of training
log_hardware_rigor()