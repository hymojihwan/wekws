import os
import logging
import re

def read_lists(list_file):
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.strip())
    return lists


def read_symbol_table(symbol_table_file):
    symbol_table = {}
    with open(symbol_table_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            symbol_table[arr[0]] = int(arr[1])
    return symbol_table


def handle_existing_log(log_file):
    if os.path.exists(log_file):
        log_dir = os.path.dirname(log_file)
        log_base = os.path.basename(log_file)
        existing_logs = [f for f in os.listdir(log_dir) if f.startswith('train') and f.endswith('.log')]
        existing_logs = sorted(existing_logs, key=lambda x: int(re.search(r'train(\d+)\.log', x).group(1)) if re.search(r'train(\d+)\.log', x) else 1, reverse=True)

        for log in existing_logs:
            match = re.search(r'train(\d+)\.log', log)
            if match:
                number = int(match.group(1))
                new_name = f"train{number+1}.log"
            else:
                new_name = "train2.log"
            os.rename(os.path.join(log_dir, log), os.path.join(log_dir, new_name))
        
        
def setup_logging(log_file, rank):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handle_existing_log(log_file)

    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Only rank 0 writes to log file
    if rank == 0:
        # File handler
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(file_formatter)

        # Stream handler (console)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        stream_handler.setFormatter(stream_formatter)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        # Force flush on file handler
        file_handler.flush()

        logging.info("Logging initialized successfully.")
    else:
        # Suppress logging for non-rank 0
        logger.addHandler(logging.NullHandler())