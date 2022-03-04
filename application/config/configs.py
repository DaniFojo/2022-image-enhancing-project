import os


class DefaultConfig:
    def __init__(self, env='development', debug=False, testing=False):
        self.env = env
        self.debug = debug
        self.testing = testing
        self.secret_key = os.urandom(32)

    def as_flask_config_dict(self):
        return {
            'ENV': self.env,
            'DEBUG': self.debug,
            'TESTING': self.testing,
            'SECRET_KEY': self.secret_key
        }


class DevConfig(DefaultConfig):
    def __init__(self):
        super().__init__(
            env='development',
            debug=True,
            testing=True
        )


class ProdConfig(DefaultConfig):
    def __init__(self):
        super().__init__(
            env='production',
            debug=False,
            testing=False)
