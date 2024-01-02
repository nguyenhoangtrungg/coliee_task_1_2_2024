import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--mode", help="Running mode", default="None")
parser.add_argument("--pretrained_model", help="Pretrain model", default="None")
parser.add_argument("--freeze_mode", help="Freeze mode", default="None")
parser.add_argument("--checkpoint_path", help="Path checkpoint", default="None")

parser.add_argument("--batch_size", help="Batch size for training", default=8, type=int)
parser.add_argument("--num_epochs", help="Number of epoch for training", default=10, type=int)
parser.add_argument("--used_gpu", help="Number of gpu for training", default=0, type=int)
parser.add_argument("--w_loss", help="Weight loss", default=0, type=float)

parser.add_argument("--training_path", help="Path to training data", default="None")
parser.add_argument("--testing_path", help="Path to testing data", default="None")
parser.add_argument("--label_path", help="Path to label data", default="None")
parser.add_argument("--csv_training_data_path", help="Path to csv training data", default="None")
parser.add_argument("--csv_testing_data_path", help="Path to csv testing data", default="None")
parser.add_argument("--negative_mode", help="Negative mode", default="None")
parser.add_argument("--negative_num", help="Number of negative sample", default=5, type=int)
parser.add_argument("--fast_dev_run", help="Fast dev run", default=0)

args = parser.parse_args()

MODE = args.mode
PRETRAIN_MODEL = args.pretrained_model
FREEZE_MODE = args.freeze_mode
CHECKPOINT = args.checkpoint_path

BATCH_SIZE = args.batch_size
N_EPOCH = args.num_epochs
USED_GPU = args.used_gpu
W_LOSS = args.w_loss

TRAINING_PATH = args.training_path
TESTING_PATH = args.testing_path
LABEL_PATH = args.label_path
CSV_TRAINING_DATA_PATH = args.csv_training_data_path
CSV_TESTING_DATA_PATH = args.csv_testing_data_path
NEGATIVE_MODE = args.negative_mode
NEGATIVE_NUM = args.negative_num

FAST_DEV_RUN = args.fast_dev_run