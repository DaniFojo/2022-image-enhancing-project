class DefaultConfig:
    def __init__(self, env='development', debug=False, testing=False):
        self.env = env
        self.debug = debug
        self.testing = testing

    def as_flask_config_dict(self):
        return {
            'ENV': self.env,
            'DEBUG': self.debug,
            'TESTING': self.testing,
        }

class DevConfig(DefaultConfig):
    def __init__(self):
        super().__init__(
            env = 'development',
            debug = True,
            testing = True
        )

class ProdConfig(DefaultConfig):
    def __init__(self):
        super().__init__(
            env = 'production',
            debug = False,
            testing = False
        )