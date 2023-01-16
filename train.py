import time
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Flickr

from ResGCN import *
from StepModel import *

parser = argparse.ArgumentParser()
parser.add_argument("--no-cuda", action="store_true", default=False,
                    help="Disable CUDA training.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--epochs", type=int, default=1000,
                    help="Number of epochs to train.")
parser.add_argument("--lr", type=float, default=1e-2,
                    help="Initial learning rate.")
parser.add_argument("--weight_decay", type=float, default=5e-4,
                    help="Weight decay (L2 loss on parameters).")
parser.add_argument("--dropout", type=float, default=0.5,
                    help="Dropout rate (1 - keep probability).")
parser.add_argument("--patience", type=int, default=100,
                    help="Patience for early stopping.")
parser.add_argument("--nhid", type=int, default=16,
                    help="Number of hidden units.")
parser.add_argument("--grcn", type=str, default="none",
                    help="Residual type for pre-model.")
parser.add_argument("--depth", type=int, default=2,
                    help="Depth for pre-model.")
parser.add_argument("--resblock", type=str, default="none",
                    help="Residual type for ResBlocks.")
parser.add_argument("--step", type=int, default=0,
                    help="Number of steps.")
parser.add_argument("--dataset", type=str, default="cora",
                    help="Type of dataset.")
parser.add_argument("--gpu", type=str, default=0,
                    help="GPU to use.")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.seed:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

def main():
    if args.dataset == "cora":
        dataset = Planetoid(root="./data/Planetoid", name="Cora")
    elif args.dataset == "citeseer":
        dataset = Planetoid(root="./data/Planetoid", name="CiteSeer")
    elif args.dataset == "pubmed":
        dataset = Planetoid(root="./data/Planetoid", name="PubMed")
    elif args.dataset == "flickr":
        dataset = Flickr(root="./data/Flickr")
    else:
        raise ValueError("Dataset not found.")

    device = "cuda:" + args.gpu if args.cuda else "cpu"
    data = dataset[0].to(device)

    for step in tqdm(range(args.step + 1)):
        if step:
            model = StepModel(pre_model=model,
                        nfeat=dataset.num_features,
                        dropout=args.dropout,
                        type=args.s_type).to(device)
            lr /= 10
        else:
            model = ResGCN(nfeat=dataset.num_features,
                        nhid=args.nhid,
                        nclass=dataset.num_classes,
                        depth=args.depth,
                        dropout=args.dropout,
                        type=args.r_type).to(device)
            lr = args.lr
        
        # training
        t_begin = time.time()
        print("Dataset:" + args.dataset, "\t",
            "Depth:", args.depth, "\t",
            "Step:", step, "\t",
            "Residual type:", args.r_type, "\t",
            "Step type:", args.s_type, "\t")
        print("Optimization Start!")
        model.update(data, args.epochs, lr, args.weight_decay, args.patience)
        print("Optimization Finished!")
        print(f"Total time elapsed: {time.time() - t_begin:.4f}s")

        # testing
        model.test(data)

if __name__ == "__main__":
    main()
