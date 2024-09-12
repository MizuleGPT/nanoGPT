import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

num_proc = 8
num_proc_load_dataset = num_proc
enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # Load only the first 32,000 rows of the dataset
    dataset = load_dataset("dustinwloring1988/fineweb-edu-sample-10BT", split="train[:32000]", num_proc=num_proc_load_dataset)
    
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
