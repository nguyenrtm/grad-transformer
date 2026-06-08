import os
import argparse
import random
import pickle
from dataset import CustomDataset

def random_split(
        length, 
        num_splits,
        save_path,
        train_test_split=0.9, 
        seed=42):
    random.seed(seed)
    indices = list(range(length))
    random.shuffle(indices)
    split_size = length // num_splits

    splits = []
    for i in range(num_splits):
        start = i * split_size
        end = (i + 1) * split_size if i < num_splits - 1 else length
        splits.append(indices[start:end])
    
    idx_dict = {}
    for i in range(len(splits)):
        idx_dict[i] = {
            'train': splits[i][:int(train_test_split * len(splits[i]))],
            'test': splits[i][int(train_test_split * len(splits[i])):]
        }
    
        delta_save_path = f"{save_path}_{i}.pkl"
        os.makedirs(os.path.dirname(delta_save_path), exist_ok=True)
        with open(delta_save_path, 'wb') as file:
            pickle.dump(idx_dict[i], file)

    return idx_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_path',
        type=str,
    )

    parser.add_argument(
        '--num_splits',
        type=int,
    )

    parser.add_argument(
        '--train_test_split',
        type=float,
    )

    parser.add_argument(
        '--save_path',
        type=str,
    )

    args = parser.parse_args()
    args_dict = vars(args)

    ds = CustomDataset(path=args.dataset_path)
    train_dataset, val_dataset, test_dataset = ds.split_dataset()
    length = len(train_dataset)
    random_split(length, train_test_split=args.train_test_split, num_splits=args.num_splits, save_path=args.save_path)