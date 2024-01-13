import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--pretrained_model", help="Pretrain model", default="bert-base-multilingual-cased")
parser.add_argument("--freeze_mode", help="Freeze mode", default="None")

parser.add_argument("--batch_size", help="Batch size for training", default=8, type=int)
parser.add_argument("--num_epochs", help="Number of epoch for training", default=10, type=int)
parser.add_argument("--used_gpu", help="Number of gpu for training", default=0, type=int)
parser.add_argument("--w_loss", help="Weight loss", default=0, type=float)
parser.add_argument("--learning_rate", help="Learning rate", default=5e-5, type=float)

parser.add_argument("--training_path", help="Path to training data", default="None")
parser.add_argument("--n_samples", help="Number of samples", default=-1, type=int)


args = parser.parse_args()

PRETRAIN_MODEL = args.pretrained_model
FREEZE_MODE = args.freeze_mode

BATCH_SIZE = args.batch_size
N_EPOCH = args.num_epochs
USED_GPU = args.used_gpu
W_LOSS = args.w_loss
LEARNING_RATE = args.learning_rate

TRAINING_PATH = args.training_path
N_SAMPLES = args.n_samples