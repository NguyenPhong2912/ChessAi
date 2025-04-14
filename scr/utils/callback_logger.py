import logging

class CallbackLogger:
    """
    A simple logger class to log training callbacks and messages.
    """

    def __init__(self, log_file: str = 'training.log'):
        """
        Initialize the callback logger.

        Args:
            log_file (str): The file to store logs.
        """
        self.logger = logging.getLogger("CallbackLogger")
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        # To avoid duplicate handlers if logger is re-instantiated
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)

    def log(self, message: str, level: str = "info"):
        """
        Log a message with a given level.

        Args:
            message (str): The message to log.
            level (str): The logging level ('info', 'warning', 'error', 'debug').
        """
        level = level.lower()
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "debug":
            self.logger.debug(message)
        else:
            self.logger.info(message)

    def log_epoch(self, epoch: int, total_reward: float, epsilon: float):
        """
        Log the details for a training epoch.

        Args:
            epoch (int): The epoch number.
            total_reward (float): Total reward for the epoch.
            epsilon (float): Current exploration rate.
        """
        message = f"Epoch {epoch}: Total Reward = {total_reward}, Epsilon = {epsilon}"
        self.log(message, level="info")
