import logging
import logging.handlers
import os
import wandb


root_logger = logging.getLogger()

# Some libraries attempt to add their own root logger handlers. This is
# getting rid of those
for handler in list(root_logger.handlers):
    root_logger.removeHandler(handler)

# Choose log format
logfmt_str = "%(asctime)s %(levelname)-8s pid:%(process)d %(name)s:%(lineno)03d:%(funcName)s    %(message)s"

# Create a console handler
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.DEBUG)
c_format=logging.Formatter(logfmt_str)
c_handler.setFormatter(c_format)

# Create a file handler
os.makedirs(os.path.join(wandb.config.RUNS_FOLDER_PTH,wandb.config.RUN_NAME), exist_ok=True)
log_pth=os.path.join(wandb.config.RUNS_FOLDER_PTH,wandb.config.RUN_NAME,'log.txt')
f_handler = logging.FileHandler(log_pth)
f_handler.setLevel(logging.DEBUG)
f_format=logging.Formatter(logfmt_str)
f_handler.setFormatter(f_format)

# Add handlers to logger
root_logger.addHandler(c_handler)
root_logger.addHandler(f_handler)