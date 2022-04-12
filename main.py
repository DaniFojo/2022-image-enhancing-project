from utils import create_directories
from train_join import training_join
from train_split import training_split

N_EPOCHS = 20
DECOM_NET_LR = 0.001
RELIGHT_NET_LR = 0.001
MODE = 'split'

if __name__ == "__main__":
    create_directories()

    if MODE == 'split':
        training_split(N_EPOCHS, DECOM_NET_LR, RELIGHT_NET_LR)
    elif MODE == 'join':
        training_join(N_EPOCHS, DECOM_NET_LR, RELIGHT_NET_LR)
    else:
        print ("error")