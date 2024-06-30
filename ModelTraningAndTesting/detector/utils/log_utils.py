import logging
import os
from datetime import datetime


def setup_logger(caller: str, log_dir: str = './logs') -> None:
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    now = datetime.now()
    now_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    filename = f'{caller}_{now_str}.log'
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{log_dir}/{filename}"),
            logging.StreamHandler()
        ]
    )
