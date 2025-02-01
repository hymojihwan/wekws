import os
import torch
import logging
import re
import yaml
import argparse
import shutil
from glob import glob

def get_args():
    parser = argparse.ArgumentParser(description="10best Average Model Creator")
    parser.add_argument("--model_dir", required=True, help="Directory containing model checkpoints")
    parser.add_argument("--output_path", default="10best_avg.pt", help="Output path for the averaged model")
    parser.add_argument("--top_k", type=int, default=10, help="Number of best checkpoints to average")
    return parser.parse_args()

def read_cv_loss(yaml_file):
    """
    Read cv_loss from the given YAML file.
    Args:
        yaml_file (str): Path to the YAML file.
    Returns:
        float: cv_loss value if available, otherwise None.
    """
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    return data.get("cv_loss", None)

def average_checkpoints(model_dir, output_path, top_k=10):
    """
    Average the top-k best checkpoints in model_dir based on cv_loss and save the averaged model.
    Args:
        model_dir (str): Directory containing the model checkpoints and YAML files.
        output_path (str): Path to save the averaged model.
        top_k (int): Number of best checkpoints to average.
    Returns:
        used_checkpoints (list): List of checkpoints used for averaging.
    """
    # Find all checkpoint (.pt) files
    checkpoints = glob(os.path.join(model_dir, "*.pt"))
    checkpoint_info = []

    for ckpt in checkpoints:
        # Assume corresponding YAML file has the same name but with .yaml extension
        yaml_file = re.sub(r'\.pt$', '.yaml', ckpt)
        if os.path.exists(yaml_file):
            cv_loss = read_cv_loss(yaml_file)
            if cv_loss is not None:
                checkpoint_info.append((ckpt, cv_loss))

    # Sort checkpoints by cv_loss (lower is better)
    checkpoint_info.sort(key=lambda x: x[1])

    # Select top-k checkpoints
    top_checkpoints = checkpoint_info[:top_k]
    used_checkpoints = [ckpt[0] for ckpt in top_checkpoints]

    # Average model weights
    logging.info(f"Averaging the following checkpoints: {used_checkpoints}")
    avg_state_dict = None
    for ckpt_path, _ in top_checkpoints:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if avg_state_dict is None:
            # Initialize avg_state_dict as float to prevent type mismatch
            avg_state_dict = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in state_dict.items()}
        for k, v in state_dict.items():
            avg_state_dict[k] += v.float() / top_k  # Ensure v is converted to float

    # Save averaged model
    torch.save(avg_state_dict, output_path)
    logging.info(f"Averaged model saved to {output_path}")

    return used_checkpoints
    
def cleanup_checkpoints(model_dir, keep_checkpoints):
    """
    Delete all checkpoints in model_dir except the ones in keep_checkpoints.
    Also preserves the latest.yaml file.
    
    Args:
        model_dir (str): Directory containing the model checkpoints.
        keep_checkpoints (list): List of checkpoints to keep.
    """
    config_path = os.path.join(model_dir, "config.yaml")
    latest_yaml = os.path.join(model_dir, "latest.yaml")
    
    # 🔥 최신 config 및 latest.yaml 유지
    keep_checkpoints.append(config_path)
    keep_checkpoints.append(latest_yaml)
    
    # 🔥 .pt 체크포인트 및 .yaml 정보도 유지해야 함!
    keep_yaml_files = [re.sub(r'\.pt$', '.yaml', ckpt) for ckpt in keep_checkpoints]
    keep_checkpoints.extend(keep_yaml_files)

    # ✅ 삭제 전에 유지할 체크포인트 확인
    logging.info(f"Keeping checkpoints: {keep_checkpoints}")

    # 🚀 모든 체크포인트(.pt) 확인 후 삭제
    all_checkpoints = glob(os.path.join(model_dir, "*.pt"))
    for ckpt in all_checkpoints:
        if ckpt not in keep_checkpoints:
            os.remove(ckpt)
            logging.info(f"Deleted checkpoint: {ckpt}")

    # 🚀 Top-10 관련 YAML 파일 유지, 나머지는 삭제
    all_yaml_files = glob(os.path.join(model_dir, "*.yaml"))
    for yaml_file in all_yaml_files:
        if yaml_file not in keep_checkpoints and yaml_file != latest_yaml:
            os.remove(yaml_file)
            logging.info(f"Deleted YAML file: {yaml_file}")

def copy_latest_yaml(model_dir, output_path):
    """
    Copy latest.yaml to match the averaged model file.
    
    Args:
        model_dir (str): Path to the model directory.
        output_path (str): Path of the averaged model file.
    """
    latest_yaml = os.path.join(model_dir, "latest.yaml")
    output_yaml = re.sub(r"\.pt$", ".yaml", output_path)
    
    if os.path.exists(latest_yaml):
        shutil.copy(latest_yaml, output_yaml)
        logging.info(f"Copied {latest_yaml} to {output_yaml}")
    else:
        logging.warning("latest.yaml not found. Skipping copy.")

def main():
    # Parse arguments
    args = get_args()

    model_dir = args.model_dir
    output_path = os.path.join(model_dir, args.output_path)
    top_k = args.top_k

    # Average the top-k best checkpoints
    used_checkpoints = average_checkpoints(model_dir, output_path, top_k=top_k)

    # Keep only the used checkpoints and delete the rest
    cleanup_checkpoints(model_dir, keep_checkpoints=used_checkpoints + [output_path])

    # Ensure latest.yaml is copied for metadata preservation
    copy_latest_yaml(model_dir, output_path)

    logging.info("10best averaging and cleanup complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()