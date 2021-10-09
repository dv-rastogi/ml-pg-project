import argparse

parser = argparse.ArgumentParser()

# Add arguments.
parser.add_argument('--num_workers', type = int, default=8, help = "Number of workers "
                                                                    "for loading the data ")
parser.add_argument('--model_save', type=str, default='model.pt',
                     help="model save for encoder")
parser.add_argument('--regression', action='store_true',
                    help = "whether the task is regression or classification")
parser.add_argument('--num_features', type=int, default=82,
                    help="number of features for each stock")
parser.add_argument('--num_days', type=int, default=60,
                    help="number of days considered for the decision of buying or selling stock")
parser.add_argument('--target_market', type=str, default="NASDAQ",
                    help="number of markets considered for the decision of buying or selling stock")
parser.add_argument('--all_markets', action='store_true',
                    help="number of markets considered for the decision of buying or selling stock")

parser.add_argument('--learning_rate', type=float, default=0.001,
                    help="starting learning rate")
parser.add_argument('--beta_1', type=float, default=0.9,
                    help="default beta_1 val for adam")
parser.add_argument('--beta_2', type=float, default=0.999,
                    help="default beta_2 val for adam")
parser.add_argument('--weight_decay', type=int, default=0.0001,
                    help="weight decay for training")                 

parser.add_argument('--batch_size', type=int, default=128,
                    help="batch size for training")
parser.add_argument('--num_epochs', type=int, default=15,
                    help="flag to indicate the final epoch of training")

FLAGS = parser.parse_args()