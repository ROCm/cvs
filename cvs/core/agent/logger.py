'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import logging

AGENT_LOGGER_NAME = "cvs.agent"
_RANK_LOG_ATTR = "cvs_rank_log"


def agent_logger():
    """The logger verbose_log emits on. A rank file handler is attached separately."""
    return logging.getLogger(AGENT_LOGGER_NAME)


def rank_log_enabled():
    """True once this process is writing {run_dir}/agent/rankN.log."""
    return any(getattr(handler, _RANK_LOG_ATTR, False) for handler in agent_logger().handlers)


def enable_rank_log(path):
    """Write cvs.agent DEBUG records to a per-rank file.

    The logger level is DEBUG so records are created while the root logger is
    still at WARNING. propagate stays on: during pytest those records also reach
    pytest's handlers, and --log-level decides whether they show on the terminal.
    """
    logger = agent_logger()
    logger.setLevel(logging.DEBUG)
    if rank_log_enabled():
        return
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    setattr(handler, _RANK_LOG_ATTR, True)
    logger.addHandler(handler)


def disable_rank_log():
    """Close the rank file and return cvs.agent to NOTSET."""
    logger = agent_logger()
    for handler in list(logger.handlers):
        if getattr(handler, _RANK_LOG_ATTR, False):
            logger.removeHandler(handler)
            handler.close()
    logger.setLevel(logging.NOTSET)
