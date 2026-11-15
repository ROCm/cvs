'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

``ConfigError``, raised by the ``cvs.config_validation`` loaders and schemas
when a config file does not match its suite's schema.
'''


class ConfigError(ValueError):
    """A config file that does not match its suite's schema.

    The message has one ``location: problem`` line per error and never the
    offending values, which can be credentials.
    """

    def __init__(self, config_path, problems):
        self.problems = problems
        super().__init__(f"{config_path} is not a valid config:\n" + "\n".join(f"  {p}" for p in problems))
