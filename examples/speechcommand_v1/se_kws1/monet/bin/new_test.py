import os
import logging
import argparse
import torch
import torch.nn.functional as F
import json
import torchaudio
import yaml
from torchmetrics.audio import SignalDistortionRatio
from pesq import pesq
from pystoi import stoi
from monet.model.se_kws_model import SEKWSModel
from monet.utils.checkpoint import load_checkpoint

def get_args():
    parser = argparse.ArgumentParser(description="Testing Speech Enhancement model")
    parser.add_argument("--config", required=True, help="Path to the config file")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--test_data", required=True, help="Path to the directory containing test data")
    parser.add_argument("--output_dir", default="results", help="Directory to save enhanced audio")
    parser.add_argument("--log_file", default="test.log", help="File to save the test log")
    args = parser.parse_args()
    return args

def acc_frame(
    logits: torch.Tensor,
    target: torch.Tensor,
):
    if logits is None:
        return 0
    pred = logits.max(1, keepdim=True)[1]
    correct = pred.eq(target.long().view_as(pred)).sum().item()
    return correct * 100.0 / logits.size(0)


def get_new_log_file(log_file):
    """
    🔥 기존 로그 파일이 있다면 test_1.log, test_2.log 형태로 새로운 파일 생성  
    """
    base_name, ext = os.path.splitext(log_file)
    count = 1

    while os.path.exists(log_file):
        log_file = f"{base_name}_{count}{ext}"
        count += 1

    return log_file


def setup_logging(log_file):
    """
    🔥 로그 파일이 존재하면 새로운 파일명으로 변경 후 저장  
    """
    log_file = get_new_log_file(log_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"✅ Logging to {log_file}")

def load_test_data(test_data_path):
    """
    Load test data from a JSON-formatted list file.
    Args:
        test_data_path (str): Path to the data.list file.
    Returns:
        List of dictionaries containing "key", "clean", and "noisy" paths.
    """
    test_data = []
    with open(test_data_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())  # JSON 파싱
                test_data.append(entry)
            except json.JSONDecodeError:
                logging.warning(f"Skipping invalid line in test data: {line.strip()}")
    return test_data

def main():
    args = get_args()

    # Set up logging
    log_path = os.path.join(args.output_dir, args.log_file)
    setup_logging(log_path)

    # Load configuration
    with open(args.config, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # Initialize model
    model = SEKWSModel(configs['se_model'], configs['kws_model'])
    model.eval()

    # Load trained checkpoint
    logging.info(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(model, args.checkpoint)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data (Modified)
    test_data = load_test_data(args.test_data)
    total_correct = 0
    total_samples = 0

    for entry in test_data:
        test_key = entry["key"]
        clean_path = entry["clean"]
        noisy_path = entry["noisy"]
        label = entry['txt']

        # Load noisy and clean signals
        noisy_signal, sample_rate = torchaudio.load(noisy_path)
        clean_signal, _ = torchaudio.load(clean_path)

        # Forward pass through the model
        with torch.no_grad():
            enhanced_signal = model.se_model(noisy_signal)
            enhanced_spec = model.spec_transform(enhanced_signal)
            logits, _ = model.kws_model(enhanced_spec.permute(0, 2, 1))
            
            target = torch.tensor(label).long()  # ✅ 정수 변환
            total_correct += acc_frame(logits, target)
            total_samples += 1

    avg_acc = (total_correct / total_samples)

    # Log and save overall results
    logging.info(f"Acc: {avg_acc:.2f}")

    # # Save overall results
    # results_path = os.path.join(args.output_dir, "results.txt")
    # with open(results_path, "w") as f:
    #     for test_key, metric_results in results:
    #         f.write(f"File: {test_key}, Metrics: {metric_results}\n")

    # logging.info(f"Evaluation complete. Results saved to {results_path}")

if __name__ == "__main__":
    main()