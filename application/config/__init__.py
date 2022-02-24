from .configs import DefaultConfig, DevConfig, ProdConfig


env = 'DEV'

if env == 'DEV':
    config = DevConfig()

elif env == 'PROD':
    config = ProdConfig()

else:
    config = DefaultConfig()

