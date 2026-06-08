import os
import random
from dataset import CustomDataset
from tqdm import tqdm

def gen_idx(subset_size, indices_list, num_sets):
    subset_indices = []
    for _ in tqdm(range(num_sets)):
        idx = random.sample(indices_list, subset_size)
        subset_indices.append(idx)
    
    return subset_indices

if __name__ == "__main__":
    import argparse
    import pickle as pkl
    
    parser = argparse.ArgumentParser(description="Generate indices for subsets of data.")
    
    parser.add_argument(
        '--subset_size',
        type=int,
        required=True,
        help="Size of each subset"
    )

    parser.add_argument(
        '--initial_idx_path',
        nargs='+',
        type=str,
        help="Use in case of multiple clients"
    )

    parser.add_argument(
        '--save_fname',
        type=str,
    )
    
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
    )
    
    parser.add_argument(
        '--num_sets',
        type=int,
        required=True,
        help="Number of subsets to generate."
    )

    parser.add_argument(
        '--split',
        type=float,
        default=1.0
    )

    parser.add_argument(
        '--single_client',
        action='store_true',
    )
    
    args = parser.parse_args()
    dataset = CustomDataset(path=args.dataset_path)
    train_dataset, val_dataset, test_dataset = dataset.split_dataset()
    split = args.split

    if args.single_client:
        # Single client
        print(f"Total number of samples to choose from: {int(len(train_dataset)*split)}")
        
        indices = gen_idx(args.subset_size, range(int(len(train_dataset)*split)), args.num_sets)

        safe_dataset_path = args.dataset_path.split('/')[1]
        os.makedirs(os.path.dirname('indices/'), exist_ok=True)
        if split != 1.0:
            with open(f'indices/split_{split}_{safe_dataset_path}_{args.subset_size}_{args.num_sets}.pkl', 'wb') as f:
                pkl.dump(indices, f)
        else:
            with open(f'indices/{safe_dataset_path}_{args.subset_size}_{args.num_sets}.pkl', 'wb') as f:
                pkl.dump(indices, f)
    else:
        # Multiple clients
        if args.initial_idx_path:
            idx_lst = []
            for idx_path in args.initial_idx_path:
                with open(idx_path, 'rb') as f:
                    indices = pkl.load(f)
                    idx_lst += indices['train']
                    idx_lst += indices['test']
                    
            print(f"Using predefined paths")
            print(f"Choosing from {len(idx_lst)} samples")
            indices = gen_idx(args.subset_size, idx_lst, args.num_sets)

            save_path = f"indices/{args.save_fname}"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pkl.dump(indices, f)