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
    parser.add_argument("--output_dir", required=True, help="Path to the directory save enhanced test data")
    parser.add_argument("--noisy_data", required=True, help="name of noisy test data")
    args = parser.parse_args()
    return args


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

    # Load configuration
    with open(args.config, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # Initialize model
    model = init_model(configs["model"])
    model.eval()

    # Load trained checkpoint
    logging.info(f"Loading checkpoint: {args.checkpoint}")
    se_checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(se_checkpoint, strict=False)
    # load_checkpoint(model, args.checkpoint)

    # Load test data (Modified)
    test_data = load_test_data(args.test_data)

    for entry in test_data:
        test_key = entry["key"]
        noisy_path = entry["noisy"]
        
        # Load noisy and clean signals
        noisy_signal, sample_rate = torchaudio.load(noisy_path)

        # Forward pass through the model
        with torch.no_grad():
            enhanced_signal = model(noisy_signal)
            
        # Save enhanced audio
        relative_path = noisy_path.split(args.noisy_data, 1)[-1].lstrip("/")
        enhanced_wav_path = os.path.join(args.output_dir, relative_path)
        # enhanced_wav_path = noisy_path.replace(args.noisy_data, f"enhanced_{args.noisy_data}")
        os.makedirs(os.path.dirname(enhanced_wav_path), exist_ok=True)
        torchaudio.save(enhanced_wav_path, enhanced_signal, sample_rate)


if __name__ == "__main__":
    main()