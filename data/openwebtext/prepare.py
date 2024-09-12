import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset
import sys

num_proc = 8
enc = tiktoken.get_encoding("gpt2")

MAX_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB in bytes

def get_size(obj):
    return sys.getsizeof(obj)

if __name__ == '__main__':
    total_size = 0
    collected_data = []
    
    # Load and process the dataset in chunks
    for chunk in load_dataset("dustinwloring1988/fineweb-edu-sample-10BT", split="train", streaming=True):
        chunk_size = get_size(chunk['text'])
        if total_size + chunk_size > MAX_SIZE_BYTES:
            break
        collected_data.append(chunk)
        total_size += chunk_size
        
        print(f"Current data size: {total_size / (1024 * 1024):.2f} MB", end='\r')
        
        if total_size >= MAX_SIZE_BYTES:
            break

    print(f"\nLoaded approximately {total_size / (1024 * 1024):.2f} MB of data")

    # Create a dataset from the collected data
    dataset = Dataset.from_list(collected_data)

    # Create a smaller validation split
    split_dataset = dataset.train_test_split(test_size=0.1, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')  # rename the test split to val

    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out

    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    print(f"Train set size: {len(tokenized['train'])} samples")
    print(f"Validation set size: {len(tokenized['val'])} samples")
    
    # Print total tokens in each set
    print(f"Total tokens in train set: {sum(tokenized['train']['len'])}")
    print(f"Total tokens in validation set: {sum(tokenized['val']['len'])}")
