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
from monet.model.se_model import init_model
from monet.utils.checkpoint import load_checkpoint

def get_args():
    parser = argparse.ArgumentParser(description="Testing Speech Enhancement model")
    parser.add_argument("--config", required=True, help="Path to the config file")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--test_data", required=True, help="Path to the directory containing test data")
    parser.add_argument("--output_dir", default="results", help="Directory to save enhanced audio")
    parser.add_argument("--metrics", nargs="+", default=["SDR", "SI-SNR"], help="Metrics to evaluate")
    parser.add_argument("--log_file", default="test.log", help="File to save the test log")
    args = parser.parse_args()
    return args


def setup_logging(log_file):
    """
    Set up logging to file and console.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def evaluate_metrics(clean, enhanced, sample_rate, metrics):
    results = {}
    if "SDR" in metrics:
        sdr_metric = SignalDistortionRatio()
        results["SDR"] = sdr_metric(enhanced, clean).item()
    if "SI-SNR" in metrics:
        si_snr_value = si_snr(enhanced, clean)
        results["SI-SNR"] = si_snr_value.item()
    if "PESQ" in metrics:
        pesq_score = pesq(sample_rate, clean.squeeze().numpy(), enhanced.squeeze().numpy(), "nb")
        results["PESQ"] = pesq_score
    if "STOI" in metrics:
        stoi_score = stoi(clean.squeeze().numpy(), enhanced.squeeze().numpy(), sample_rate, extended=False)
        results["STOI"] = stoi_score
    return results


def si_snr(predicted, target, eps=1e-8):
    """
    Compute Scale-Invariant Signal-to-Noise Ratio (SI-SNR).
    """
    target_norm = torch.sum(target * predicted, dim=-1, keepdim=True) / (torch.sum(target ** 2, dim=-1, keepdim=True) + eps)
    target_proj = target_norm * target
    noise = predicted - target_proj
    si_snr_value = 10 * torch.log10((torch.sum(target_proj ** 2, dim=-1) + eps) / (torch.sum(noise ** 2, dim=-1) + eps))
    return torch.mean(si_snr_value)

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
    model = init_model(configs["model"])
    model.eval()

    # Load trained checkpoint
    logging.info(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(model, args.checkpoint)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data (Modified)
    test_data = load_test_data(args.test_data)
    results = []

    # Metric lists for averaging
    sdr_list = []
    si_snr_list = []

    for entry in test_data:
        test_key = entry["key"]
        clean_path = entry["clean"]
        noisy_path = entry["noisy"]

        # Load noisy and clean signals
        noisy_signal, sample_rate = torchaudio.load(noisy_path)
        clean_signal, _ = torchaudio.load(clean_path)

        # Forward pass through the model
        with torch.no_grad():
            _, enhanced_signal = model(noisy_signal)
            
        # Save enhanced audio
        enhanced_path = os.path.join(args.output_dir, f"enhanced_{test_key}.wav")
        # torchaudio.save(enhanced_path, enhanced_signal, sample_rate)

        # Compute metrics
        min_length = min(clean_signal.shape[-1], enhanced_signal.shape[-1])
        clean_signal = clean_signal[..., :min_length]
        enhanced_signal = enhanced_signal[..., :min_length]
        metric_results = evaluate_metrics(clean_signal, enhanced_signal, sample_rate, args.metrics)
        
        # logging.info(f"File: {test_key}, Metrics: {metric_results}")
        results.append((test_key, metric_results))

        # Store SDR and SI-SNR for averaging
        if "SDR" in metric_results:
            sdr_list.append(metric_results["SDR"])
        if "SI-SNR" in metric_results:
            si_snr_list.append(metric_results["SI-SNR"])

    # Calculate average metrics
    avg_sdr = sum(sdr_list) / len(sdr_list) if sdr_list else None
    avg_si_snr = sum(si_snr_list) / len(si_snr_list) if si_snr_list else None

    # Log and save overall results
    logging.info(f"Average SDR: {avg_sdr:.4f}, Average SI-SNR: {avg_si_snr:.4f}")

    # # Save overall results
    # results_path = os.path.join(args.output_dir, "results.txt")
    # with open(results_path, "w") as f:
    #     for test_key, metric_results in results:
    #         f.write(f"File: {test_key}, Metrics: {metric_results}\n")

    # logging.info(f"Evaluation complete. Results saved to {results_path}")

if __name__ == "__main__":
    main()