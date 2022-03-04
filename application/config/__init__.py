from .configs import DevConfig


ENV = 'DEV'

if ENV == 'DEV':
    config = DevConfig()

# elif ENV == 'PROD':
#     config = ProdConfig()

# else:
#     config = DefaultConfig()
