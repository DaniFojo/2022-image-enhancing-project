import argparse


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# argument parser

parser = argparse.ArgumentParser()
#parser.add_argument("--n_samples", help="amount of samples to train with", type=int, default=1000)
args = parser.parse_args()



# load datasets



# train

