"""
Basic Retry Decorator.
Handles retry logic with exponential backoff for transient failures.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import time
import functools
import logging

from cvs.lib import globals

log = logging.getLogger(__name__)


class RetryIfEnabled:
    """
    Class-based decorator that handles retry logic with exponential backoff.

    Provides basic retry functionality for handling transient failures.
    Can be extended by subclasses to add cleanup logic.

    Usage:
        @RetryIfEnabled()
        def some_function(..., retry_config=None):
            # Function implementation
    """

    def __init__(self):
        """Initialize the decorator"""
        pass

    def __call__(self, func):
        """
        Decorator that applies retry logic if retry_config is enabled.

        Checks for 'retry_config' in function kwargs and applies retry
        if enabled.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retry_config = kwargs.get('retry_config')

            if retry_config and retry_config.get('cvs_retry_on_failure', False):
                return self._execute_with_retry(func, args, kwargs, retry_config)
            else:
                # Retry disabled - just run the function normally
                log.debug(f"Retry disabled for {func.__name__}, running once")
                return func(*args, **kwargs)

        return wrapper

    def _execute_with_retry(self, func, args, kwargs, retry_config):
        """
        Execute function with retry logic.

        Implements simple retry with cleanup after failure.
        Checks globals.error_list for failures instead of relying on exceptions.
        """
        max_retries = retry_config.get('cvs_max_retries', 1)
        retry_delay = retry_config.get('cvs_retry_delay_seconds', 0)

        test_name = kwargs.get('test_name', func.__name__)

        # max_retries means number of retry attempts, so total attempts = max_retries + 1
        total_attempts = max_retries + 1

        for attempt in range(total_attempts):
            log.info(f"Test '{test_name}' - Attempt {attempt + 1}/{total_attempts}")

            # Reset error list before each attempt
            globals.error_list = []

            # Wait before retry (except first attempt)
            if attempt > 0:
                if retry_delay > 0:
                    log.info(f"Waiting {retry_delay}s before retry...")
                    time.sleep(retry_delay)

            # Execute the test
            result = func(*args, **kwargs)

            # Check if the test failed by examining globals.error_list
            if len(globals.error_list) > 0:
                log.error(f"Test '{test_name}' failed on attempt {attempt + 1}: {len(globals.error_list)} error(s) - {globals.error_list}")

                # Perform cleanup after failure (subclass hook)
                self._cleanup_after_failure(kwargs, retry_config)

                # Re-raise if this was the last attempt
                if attempt >= total_attempts - 1:
                    log.error(f"Test '{test_name}' failed after {total_attempts} attempts")
                    raise Exception(f'Test failed after {total_attempts} attempts with {len(globals.error_list)} error(s): {globals.error_list}')
                else:
                    log.info(f"Retrying test '{test_name}'...")
            else:
                log.info(f"Test '{test_name}' succeeded on attempt {attempt + 1}")
                return result

    def _cleanup_after_failure(self, kwargs, retry_config):
        """
        Hook for cleanup after failure. Override in subclasses.

        Args:
            kwargs: Function kwargs
            retry_config: Retry configuration
        """
        pass


# Create a singleton instance for use as decorator
retry_if_enabled = RetryIfEnabled()