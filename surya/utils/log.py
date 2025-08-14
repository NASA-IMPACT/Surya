import functools
import logging
import os
import sys
from time import time
from packaging.version import Version
import wandb
from typing import Dict, Optional, Any


if Version(wandb.__version__) < Version("0.20.0"):
    WANDB_USE_SYNC = True
else:
    WANDB_USE_SYNC = False


def log(
    run,
    data: Dict[str, Any],
    step: Optional[int] = None,
    commit: Optional[bool] = None,
    sync: Optional[bool] = None,
) -> None:
    if run is not None:
        # Note: wandb changed the .log API with version 0.20.0.
        # This includes: "Removed no-op sync argument from wandb.Run::log function"
        # We didn't test whether sync has any function here. But since we did
        # all our development with it, let's keep it here for now.
        # See https://github.com/wandb/wandb/releases/tag/v0.20.0
        if WANDB_USE_SYNC:
            run.log(data, step, commit, sync)
        else:
            run.log(data, step, commit)
    else:
        print(data)


# See: https://github.com/microsoft/Swin-Transformer/blob/main/logger.py
# See: https://github.com/Meituan-AutoML/Twins/blob/main/logger.py
def create_logger(output_dir: str, dist_rank: int, name: str) -> logging.Logger:
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = "[%(asctime)s %(name)s]: %(levelname)s %(message)s"

    # create console handlers
    if name.endswith("main"):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(
        os.path.join(output_dir, f"{name}.log"), mode="a"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    return logger


def log_decorator(logger, _func=None):
    def log_decorator_info(func):
        @functools.wraps(func)
        def log_decorator_wrapper(*args, **kwargs):
            """Create a list of the positional arguments passed to function.
            - Using repr() for string representation for each argument. repr() is similar to str() only
            difference being it prints with a pair of quotes and if we calculate a value we get more
            precise value than str().
            """

            # py_file_caller = getframeinfo(stack()[1][0])

            local_rank = os.environ.get("LOCAL_RANK", default=None)
            rank = os.environ.get("LOCAL_RANK", default=None)

            try:
                """log return value from the function"""
                start_time = time()
                value = func(*args, **kwargs)
                if local_rank is None or rank is None:
                    logger.info(
                        f"Function '{func.__name__}' - Execution time: {(time() - start_time):.1f} seconds."
                    )
                else:
                    logger.info(
                        f"Function '{func.__name__}' - Execution time: {(time() - start_time):.1f} "
                        f"seconds on rank {os.environ['RANK']} and local_rank {os.environ['LOCAL_RANK']}."
                    )
            except Exception as err:
                logger.error(f"Exception: {err}")
                raise
            return value

        # Return the pointer to the function
        return log_decorator_wrapper

    # Decorator was called with arguments, so return a decorator function that can read and return a function
    if _func is None:
        return log_decorator_info
    # Decorator was called without arguments, so apply the decorator to the function immediately
    else:
        return log_decorator_info(_func)
