import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm
from tqdm import trange
from model import SeqClassifier
from dataset import SeqClsDataset
from torch.utils.data import DataLoader
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }

    #print(datasets);
    # TODO: crecate DataLoader for train / dev datasets
    dataloader = {}
    dataloader[TRAIN] = DataLoader(datasets[TRAIN], batch_size = args.batch_size, shuffle=True, collate_fn = datasets[TRAIN].collate_fn)
    dataloader[DEV] = DataLoader(datasets[DEV], batch_size = args.batch_size, shuffle=True, collate_fn = datasets[DEV].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    device = args.device
    # print(args)
    model = SeqClassifier(
        embeddings = embeddings,
        hidden_size = args.hidden_size,
        num_layers =  args.num_layers,
        dropout = args.dropout,
        bidirectional = args.bidirectional,
        num_class = 150,
    ).to(args.device)

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # init Loss function
    criterion = torch.nn.CrossEntropyLoss() 
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    total_loss = 0
    acc_sum = 0
    best_acc = 0
    best_loss = 1e9
    model.train()
    early_stop = 0
    early_stop_hope = 20
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        total_loss, acc_sum = 0, 0
        for i, inputs in enumerate(dataloader[TRAIN]):
            inputs['text'] = inputs['text'].to(args.device)
            inputs['intent'] = inputs['intent'].to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            #print(len(outputs))
            #print(len(inputs['intent']))
            loss = criterion(outputs, inputs['intent'])
            loss.backward()
            #print(loss)
            prediction = torch.argmax(outputs, dim=1)
            total_correct = 0
            for i in range(len(prediction)):
                if prediction[i] == inputs['intent'][i]:
                    total_correct += 1
            
            acc_sum += (total_correct / args.batch_size)
            total_loss += loss.item()
            optimizer.step()
        
        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.eval()
        with torch.no_grad():
            total_loss, acc_sum = 0, 0
            for i, inputs in enumerate(dataloader[DEV]):
                inputs['text'] = inputs['text'].to(args.device)
                inputs['intent'] = inputs['intent'].to(args.device)
                outputs = model(inputs)
                # print(len(outputs))
                # print(len(inputs['intent']))

                loss = criterion(outputs, inputs['intent'])
                
                prediction = torch.argmax(outputs, dim=1)
                total_correct = 0
                for i in range(len(prediction)):
                    if prediction[i] == inputs['intent'][i]:
                        total_correct += 1
                
                acc_sum += (total_correct / args.batch_size)
                total_loss += loss.item()


            if acc_sum > best_acc:
                best_acc = acc_sum
                torch.save(model.state_dict(), args.ckpt_dir / 'intent_model.ckpt')
                early_stop = 0
            elif best_loss > total_loss:
                best_loss = total_loss
                early_stop = 0
            elif best_loss < total_loss:
                early_stop += 1
            if early_stop >= early_stop_hope:
                print("early stop")
                break
            
            # print(total_loss)
        model.train()
        pass
    print(best_loss)
            
        
        

    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
