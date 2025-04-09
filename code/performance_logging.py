from configs import *


class PerformanceLogger:
    def __init__(self, log_dir='logs/python_logs'):
        # Create full path and ensure it exists
        self.log_dir = f'{ROOT_DIR}/code/{log_dir}'
        os.makedirs(self.log_dir, exist_ok=True)
        self.loggers = {}
        
    def get_logger(self, dataset, name):
        """Get logger with optional algorithm type."""
        logger_name = f"{dataset}_{name}"
        
        if logger_name not in self.loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.INFO)
            logger.handlers.clear()
            
            timestamp = datetime.now().strftime('%Y%m%d')
            log_file_path = os.path.join(self.log_dir, f'{logger_name}_{timestamp}.log')
            
            # Ensure the parent directory exists
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            
            file_handler = logging.FileHandler(
                log_file_path,
                mode='w' 
            )
            console_handler = logging.StreamHandler()
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
            self.loggers[logger_name] = logger
            
        return self.loggers[logger_name]

def log_execution_time(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        execution_time = time.time() - start_time
        self.logger.info(f"{func.__name__} completed in {execution_time:.2f}s")
        return result
    return wrapper

def log_gpu_stats(logger, prefix=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logger.debug(
            f"{prefix}GPU Memory: "
            f"Allocated={allocated:.1f}MB, "
            f"Reserved={reserved:.1f}MB"
        )

# Create a singleton instance
performance_logger = PerformanceLogger()

def get_client_logger(client_id, algorithm_type=None):
    """Get client logger with optional algorithm type."""
    return performance_logger.get_logger(f"client_{client_id}", algorithm_type)

def get_server_logger(algorithm_type=None):
    """Get server logger with optional algorithm type."""
    return performance_logger.get_logger("server", algorithm_type)