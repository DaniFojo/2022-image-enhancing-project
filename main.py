import argparse
from utils import create_directories
from train_join import training_join
from train_split import training_split

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--n_epochs", help="amount of epochs to train with", type=int, default=20)
parser.add_argument("-dlr", "--decom_lr", help="learning rate for decomposition", type=float, default=0.001)
parser.add_argument("-rlr", "--rel_lr", help="learning rate for relight", type=float, default=0.001)
parser.add_argument("-m", "--mode", help="join: train decom and relight together, split: train decom and relight separately", type=str, default="split")
parser.add_argument("-i", "--ignore_ienhance", help="bool to ignore enhanced illuminance if mode is join", type=bool, default=False)
parser.add_argument("-s", "--s_epochs", help="amount of epochs to store models", type=int, default=10)
parser.add_argument("-t", "--transposed", help="use convolutional transpose", type=bool, default=False)
args = parser.parse_args()

if __name__ == "__main__":
    create_directories()

    if args.mode == 'split':
        training_split(args.n_epochs, args.decom_lr, args.rel_lr, args.s_epochs, args.transposed)
    elif args.mode == 'join':
        training_join(args.n_epochs, args.decom_lr, args.rel_lr, args.s_epochs, args.ignore_ienhance)
    else:
        print ("error")
