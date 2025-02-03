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

def read_latest_cv_losses(latest_yaml_path):
    """  
    `latest.yaml`에서 `cv_losses` 리스트를 읽어 반환  
    """
    if not os.path.exists(latest_yaml_path):
        logging.error(f"❌ latest.yaml not found at {latest_yaml_path}!")
        return None, None

    with open(latest_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    cv_losses = data.get("cv_losses", [])
    epoch = data.get("epoch", None)

    if not cv_losses or epoch is None:
        logging.error("❌ No valid `cv_losses` or `epoch` found in latest.yaml!")
        return None, None

    return cv_losses, epoch

def average_checkpoints(model_dir, output_path, top_k=10):
    """
    🔥 `latest.yaml`에서 `cv_losses`를 읽어 가장 좋은 `top_k` 개 체크포인트 평균  
    """
    latest_yaml_path = os.path.join(model_dir, "latest.yaml")
    cv_losses, total_epochs = read_latest_cv_losses(latest_yaml_path)

    if cv_losses is None:
        logging.error("❌ Cannot proceed without `cv_losses` data!")
        return []

    # 🔥 1️⃣ 모든 체크포인트(.pt) 파일 가져오기  
    checkpoints = glob(os.path.join(model_dir, "*.pt"))
    if not checkpoints:
        logging.error("❌ No checkpoint files found in model directory!")
        return []

    # 🔥 2️⃣ 최신 `epoch` 값에 따라 체크포인트 이름 예측 (`<epoch>.pt` 형태)  
    epoch_to_ckpt = {int(re.search(r"(\d+).pt$", ckpt).group(1)): ckpt for ckpt in checkpoints if re.search(r"(\d+).pt$", ckpt)}

    # 🔥 3️⃣ 최신 `cv_losses` 리스트와 `epoch`를 매칭  
    valid_checkpoints = [(epoch_to_ckpt[i + 1], cv_losses[i]) for i in range(len(cv_losses)) if (i + 1) in epoch_to_ckpt]

    if not valid_checkpoints:
        logging.error("❌ No valid checkpoints matched with cv_losses!")
        return []

    # 🔥 4️⃣ `cv_loss`가 작은 `top_k` 개 체크포인트 선택  
    valid_checkpoints.sort(key=lambda x: x[1])  # 작은 `cv_loss`가 우선  
    top_checkpoints = valid_checkpoints[:top_k]
    used_checkpoints = [ckpt[0] for ckpt in top_checkpoints]

    logging.info(f"✅ Averaging the following checkpoints: {used_checkpoints}")

    # 🔥 5️⃣ 모델 가중치 평균 계산  
    avg_state_dict = None
    for ckpt_path, _ in top_checkpoints:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if avg_state_dict is None:
            avg_state_dict = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in state_dict.items()}
        for k, v in state_dict.items():
            avg_state_dict[k] += v.float() / top_k  # 평균 계산  

    # 🔥 6️⃣ 평균 모델 저장  
    torch.save(avg_state_dict, output_path)
    logging.info(f"✅ Averaged model saved to {output_path}")

    return used_checkpoints
    
def cleanup_checkpoints(model_dir, keep_checkpoints):
    """
    🔥 `latest.yaml`과 `10best_avg.pt`만 유지하고, 나머지 `.yaml`은 삭제
    """
    config_path = os.path.join(model_dir, "config.yaml")
    latest_yaml = os.path.join(model_dir, "latest.yaml")

    # 🔥 1️⃣ 유지할 파일 리스트
    keep_checkpoints.append(config_path)  # `config.yaml` 유지
    keep_checkpoints.append(latest_yaml)  # `latest.yaml` 유지

    # 🔥 2️⃣ 불필요한 `n.pt` 체크포인트 삭제
    all_checkpoints = glob(os.path.join(model_dir, "*.pt"))
    for ckpt in all_checkpoints:
        if ckpt not in keep_checkpoints:
            os.remove(ckpt)
            logging.info(f"🗑️ Deleted checkpoint: {ckpt}")

    # 🔥 3️⃣ 모든 `.yaml` 중 `latest.yaml` 제외하고 삭제
    all_yaml_files = glob(os.path.join(model_dir, "*.yaml"))
    for yaml_file in all_yaml_files:
        if yaml_file not in keep_checkpoints and yaml_file != latest_yaml:
            os.remove(yaml_file)
            logging.info(f"🗑️ Deleted YAML file: {yaml_file}")


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