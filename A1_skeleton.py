
import torch, pickle
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from torch.utils.data import DataLoader
import numpy as np
import sys, time, os

import regex as re
###
### Part 1. Tokenization.
###

# Byte-pair encoding (BPE): the main idea
# ---------------------------------------
# A token is one piece of text (not necessarily a word/subword/"token").
# In this tokenizer, every token is stored as one
# or more bytes. A pair means two tokens that appear next to each other, and a
# merge means joining such a pair to make one longer token.
#
# Training BPE:
#   1. Split every line of the training text into individual UTF-8 bytes.
#   2. Start the vocabulary with the four special tokens and all 256 possible
#      byte values. The vocabulary maps each token to an integer ID.
#   3. Count how often each pair of neighboring tokens occurs.
#   4. Join the most common pair everywhere it occurs. Add the joined token to
#      the vocabulary and save the pair as a merge rule.
#   5. Repeat steps 3 and 4 until the requested vocabulary size is reached 
#      (NB: make sure it is provided).

#
# Encoding new text:
#   1. Remove whitespace from the beginning and end.
#   2. Split the text into individual UTF-8 bytes.
#   3. Apply the saved merge rules in the same order in which they were learned.
#   4. Replace the resulting tokens with their vocabulary IDs.
#   5. Add BOS/EOS IDs, and optionally truncate or pad the sequence.
#
# This simple version treats a whole non-empty line as one sequence. Therefore,
# it may learn tokens that contain spaces or parts of more than one word.

def split_into_bytes(text):
    """Split a string into individual UTF-8 byte tokens.

    Args:
        text: The input string to encode.

    Returns:
        A list containing one ``bytes`` object for each byte in the text.
    """
    encoded_text = text.encode("utf-8")
    # bytes_list = [encoded_text[i: i+1] for i in range(len(encoded_text))]
    bytes_list = [bytes([b]) for b in encoded_text]
    
    return bytes_list
    

def build_tokenizer(train_file, max_voc_size=None, model_max_length=None,
                    pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>'):
    """ Build a tokenizer from the given file.

        Args:
             train_file:        The name of the file containing the training texts.
             max_voc_size:      The maximally allowed size of the vocabulary.
             model_max_length:  Truncate texts longer than this length.
             pad_token:         The dummy string corresponding to padding.
             unk_token:         The dummy string corresponding to out-of-vocabulary tokens.
             bos_token:         The dummy string corresponding to the beginning of the text.
             eos_token:         The dummy string corresponding to the end the text.
    """

    if max_voc_size is None or max_voc_size < 260:
        raise ValueError("max_voc_size must be at least 260 (4 special + 256 bytes)")
    # TODO: build the vocabulary, using a byte-level BPE tokenization (explained above)
    # Then return a tokenizer object (implemented below).

    vocab = {bos_token: 0, eos_token: 1, unk_token: 2, pad_token: 3}
    for i in range(256):
        vocab.update({bytes([i]): i+4})

    with open(train_file, encoding="utf-8") as f:
        frequency_table = Counter()
        for line in f:
            line = line.strip()
            if line:
                frequency_table[tuple(split_into_bytes(line))] += 1

    pair_frequencies = Counter()
    for chars, counts in frequency_table.items():
        for i in range(len(chars) - 1):
            pair_frequencies[(chars[i], chars[i+1])] += counts

    token_to_lines = {}
    for token_tuple in frequency_table:
        for token in set(token_tuple):
            if token not in token_to_lines:
                token_to_lines[token] = set()
            token_to_lines[token].add(token_tuple)
    merges = []
    while len(vocab) < max_voc_size and pair_frequencies:
        best_pair = max(pair_frequencies.items(), key=lambda x: (x[1], x[0]))[0]
        a, b = best_pair
        merged_token = a + b
        merges.append(best_pair)
        # Different merge paths can produce the same bytes; keep the existing id then.
        if merged_token not in vocab:
            vocab[merged_token] = len(vocab)

        # Only lines containing both tokens can change.
        lines_to_update = token_to_lines.get(a, set()) & token_to_lines.get(b, set())
        touched_pairs = set()
        for old in lines_to_update:
            new = []
            i = 0
            while i < len(old):
                if i + 1 < len(old) and old[i] == a and old[i + 1] == b:
                    new.append(merged_token)
                    i += 2
                else:
                    new.append(old[i])
                    i += 1
            new = tuple(new)
            if new == old:  # both tokens present, but never adjacent
                continue

            # Replace the line's old pair counts by its new ones.
            freq = frequency_table.pop(old)
            for i in range(len(old) - 1):
                pair = (old[i], old[i + 1])
                pair_frequencies[pair] -= freq
                touched_pairs.add(pair)
            for i in range(len(new) - 1):
                pair = (new[i], new[i + 1])
                pair_frequencies[pair] += freq
                touched_pairs.add(pair)
            frequency_table[new] += freq

            for token in set(old):
                token_to_lines[token].discard(old)
                if not token_to_lines[token]:
                    del token_to_lines[token]
            for token in set(new):
                token_to_lines.setdefault(token, set()).add(new)

        for pair in touched_pairs:
            if pair_frequencies[pair] <= 0:
                pair_frequencies.pop(pair, None)

    return A1Tokenizer(vocab, merges, 
    vocab[pad_token], vocab[unk_token], vocab[bos_token], vocab[eos_token],
    model_max_length=model_max_length
    )

class A1Tokenizer:
    """A minimal implementation of a tokenizer similar to tokenizers in the HuggingFace library."""

    def __init__(self, vocab, merges, pad_token_id, unk_token_id, bos_token_id, eos_token_id, model_max_length):
        # TODO: store all values you need in order to implement __call__ below.
        self.str_to_int = vocab
        self.int_to_str = {i: t for t, i in vocab.items()}
        self.merges = merges
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.pad_token_id = pad_token_id     # Compulsory attribute.
        self.unk_token_id = unk_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.model_max_length = model_max_length # Needed for truncation.

    def _apply_merges(self, tokens):
        ranks = self.merge_ranks
        while len(tokens) > 1:
            best = min(zip(tokens, tokens[1:]), key=lambda p: ranks.get(p, len(ranks)))
            if best not in ranks:
                break
            a, b = best
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(a + b)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        """Tokenize the given texts and return a BatchEncoding containing the integer-encoded tokens.
           
           Args:
             texts:           The texts to tokenize.
             truncation:      Whether the texts should be truncated to model_max_length.
             padding:         Whether the tokenized texts should be padded on the right side.
             return_tensors:  If None, then return lists; if 'pt', then return PyTorch tensors.

           Returns:
             A BatchEncoding where the field `input_ids` stores the integer-encoded texts.
        """
        if return_tensors and return_tensors != 'pt':
            raise ValueError('Should be pt')
        
        # TODO: Your work here is to split the texts into words and map them to integer values.
        # 
        # - If `truncation` is set to True, the length of the encoded sequences should be 
        #   at most self.model_max_length.
        # - If `padding` is set to True, then all the integer-encoded sequences should be of the
        #   same length. That is: the shorter sequences should be "padded" by adding dummy padding
        #   tokens on the right side.
        # - If `return_tensors` is undefined, then the returned `input_ids` should be a list of lists.
        #   Otherwise, if `return_tensors` is 'pt', then `input_ids` should be a PyTorch 2D tensor.

        # TODO: Return a BatchEncoding where input_ids stores the result of the integer encoding.
        # Optionally, if you want to be 100% HuggingFace-compatible, you should also include an 
        # attention mask of the same shape as input_ids. In this mask, padding tokens correspond
        # to the the value 0 and real tokens to the value 1.

        ids_ = []
        for text in texts:
            ids = []
            tokens = self._apply_merges(split_into_bytes(text.strip()))
            for token in tokens:
                ids.append(self.str_to_int.get(token, self.unk_token_id))

            if truncation and self.model_max_length is not None:
                ids = ids[:self.model_max_length-2]  # Reserve space for BOS and EOS tokens

            ids_.append([self.bos_token_id] + ids + [self.eos_token_id])

        if padding:
            max_len = max(len(x) for x in ids_)
            mask = [[1]*len(x) + [0]*(max_len-len(x)) for x in ids_]
            ids_ = [x + [self.pad_token_id]*(max_len-len(x)) for x in ids_]
        else:
            mask = [[1]*len(x) for x in ids_]

        if return_tensors == 'pt':
            ids_, mask = torch.tensor(ids_), torch.tensor(mask)
        return BatchEncoding({'input_ids': ids_, 'attention_mask': mask})


    @property
    def vocab_size(self):
        return len(self.str_to_int)

    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.str_to_int)

    def decode(self, ids):
        """Return a string decoded from given input ids"""
        byte_tokens = []
        for i in ids:
            tok = self.int_to_str[int(i)]
            if isinstance(tok, bytes):
                byte_tokens.append(tok)
        return b''.join(byte_tokens).decode('utf-8', errors='replace')
    
    def save(self, filename):
        """Save the tokenizer to the given file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename):
        """Load a tokenizer from the given file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)
   

###
### Part 3. Defining the model.
###

class A1RNNModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the RNN-based language model."""
    def __init__(self, vocab_size=None, embedding_size=None, hidden_size=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

class A1RNNModel(PreTrainedModel):
    """The neural network model that implements a RNN-based language model."""
    config_class = A1RNNModelConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.rnn = nn.GRU(config.embedding_size, config.hidden_size, batch_first=True)
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size)
        self.post_init() # for compatibility 
        
    def forward(self, labels):
        """The forward pass of the RNN-based language model.
        
           Args:
             X:  The input tensor (2D), consisting of a batch of integer-encoded texts.
           Returns:
             CausalLMOutputWithPast object that stores loss and logits (the output tensor (3D) for all token positions for all vocabulary items.)
        """

        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)

        X = labels[:, :-1].clamp(min=0) # all columns in input_ids except the last one
        Y = labels[:, 1:] # all columns in input_ids except the first one

        assert(X.shape == Y.shape)

        embedded = self.embedding(X)
        rnn_out, _ = self.rnn(embedded)
        out = self.unembedding(rnn_out)

        #  compute the loss for the model output and Y 
        return CausalLMOutputWithPast(loss=loss_func(out.reshape(-1, out.size(-1)), Y.reshape(-1)), logits=out)


###
### Part 4. Training the language model.
###

## Hint: the following TrainingArguments hyperparameters may be relevant for your implementation:
#
# - optim:            What optimizer to use. You can assume that this is set to 'adamw_torch',
#                     meaning that we use the PyTorch AdamW optimizer.
# - eval_strategy:    You can assume that this is set to 'epoch', meaning that the model should
#                     be evaluated on the validation set after each epoch
# - use_cpu:          Force the trainer to use the CPU; otherwise, CUDA or MPS should be used.
#                     (In your code, you can just use the provided method select_device.)
# - learning_rate:    The optimizer's learning rate.
# - num_train_epochs: The number of epochs to use in the training loop.
# - per_device_train_batch_size: 
#                     The batch size to use while training.
# - per_device_eval_batch_size:
#                     The batch size to use while evaluating.
# - output_dir:       The directory where the trained model will be saved.

class TrainingArguments(object):

    def __init__(self, optim='adamw_torch', eval_strategy='epoch', use_cpu=False, learning_rate=5e-5,
                 num_train_epochs=3, per_device_train_batch_size=8, per_device_eval_batch_size=8,
                 output_dir='./results'):
        self.optim = optim
        self.eval_strategy = eval_strategy
        self.use_cpu = use_cpu
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.output_dir = output_dir

class A1Trainer:
    """A minimal implementation similar to a Trainer from the HuggingFace library."""

    def __init__(self, model, args, train_dataset, eval_dataset, processing_class):
        """Set up the trainer.
           
           Args:
             model:          The model to train.
             args:           The training parameters stored in a TrainingArguments object.
             train_dataset:  The dataset containing the training documents.
             eval_dataset:   The dataset containing the validation documents.
             eval_dataset:   The dataset containing the validation documents.
             processing_class:      The tokenizer.
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class

        assert(args.optim == 'adamw_torch')
        assert(args.eval_strategy == 'epoch')

    def select_device(self):
        """Return the device to use for training, depending on the training arguments and the available backends."""
        if self.args.use_cpu:
            return torch.device('cpu')
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
            
    def train(self):
        """Train the model."""
        args = self.args

        device = self.select_device()
        print('Device:', device)
        self.model.to(device)
        
        
        # TODO: Relevant arguments: at least args.learning_rate, but you can optionally also consider
        # other Adam-related hyperparameters here.
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)

        # TODO: Relevant arguments: args.per_device_train_batch_size, args.per_device_eval_batch_size
        pad_id = self.processing_class.pad_token_id
        if pad_id is None: # e.g. the GPT-2 tokenizer has no pad token
            pad_id = self.processing_class.eos_token_id

        def collate(items):
            seqs = [x['input_ids'] if isinstance(x, dict) else x for x in items]
            n = max(len(q) for q in seqs)
            return torch.tensor([list(q) + [pad_id] * (n - len(q)) for q in seqs])

        train_loader = DataLoader(self.train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(self.eval_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, collate_fn=collate)
        
        # TODO: Your work here is to implement the training loop.
        #       
        # for each training epoch (use args.num_train_epochs here):
        #   for each batch B in the training set:
        #
        #       PREPROCESSING AND FORWARD PASS:
        #       labels = apply your tokenizer to B
        #       replace padding tokens with value -100 so loss function will ignore them
	    #       put labels onto the GPU (or whatever device you use)
        #       apply the model to labels
        #
        #       BACKWARD PASS AND MODEL UPDATE:
        #       optimizer.zero_grad()
        #       loss.backward() # loss is stored as an attribute of model's output
        #       optimizer.step()

        for epoch in range(args.num_train_epochs):
            self.model.train()
            for batch in train_loader:
                labels = batch.to(device)
                labels[labels == pad_id] = -100
                outputs = self.model(labels)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Evaluate on validation set
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    labels = batch.to(device)
                    labels[labels == pad_id] = -100
                    outputs = self.model(labels)
                    val_loss += outputs.loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f'Epoch {epoch + 1}/{args.num_train_epochs}, Train Loss: {loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        print(f'Saving to {args.output_dir}.')
        self.model.save_pretrained(args.output_dir)


def perplexity(model, dataset, processing_class, batch_size=8):
    """Compute the perplexity of the model on the given dataset."""
    device = model.device
    pad_id = processing_class.pad_token_id

    def collate(items):
        seqs = [x['input_ids'] if isinstance(x, dict) else x for x in items]
        n = max(len(q) for q in seqs)
        return torch.tensor([list(q) + [pad_id] * (n - len(q)) for q in seqs])

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            labels = batch.to(device)
            labels[labels == pad_id] = -100
            outputs = model(labels)
            loss = outputs.loss.item()
            total_loss += loss * (labels != -100).sum().item()
            total_tokens += (labels != -100).sum().item()

    avg_loss = total_loss / total_tokens
    return np.exp(avg_loss)

if __name__ == '__main__':
    if not os.path.exists("tokenizer.pkl"):
        print("Building tokenizer...")
        tokenizer = build_tokenizer('train.txt', max_voc_size=10000, model_max_length=128)
        tokenizer.save('tokenizer.pkl')
    else:
        print("Loading tokenizer...")
        tokenizer = A1Tokenizer.from_file('tokenizer.pkl')
    test_texts = ['This is a test.', 'Another test.']
    tokenizer(test_texts, return_tensors='pt', padding=True,
          truncation=True)

    from datasets import load_dataset
    dataset = load_dataset('text', data_files={'train': "train.txt", 'val': "val.txt"})
    dataset = dataset.filter(lambda x: x['text'].strip() != '')
    
    dataset = dataset.map(
    lambda batch: dict(tokenizer(batch['text'], truncation=True, padding=False)),
    batched=True,
    remove_columns=['text'],
    num_proc=8
    )

    # start training
    model_config = A1RNNModelConfig(vocab_size=tokenizer.vocab_size, embedding_size=128, hidden_size=256)
    model = A1RNNModel(model_config)
    args = TrainingArguments(learning_rate=5e-4, num_train_epochs=40, per_device_train_batch_size=512, per_device_eval_batch_size=512, output_dir='./results')
    trainer = A1Trainer(model, args, dataset['train'], dataset['val'], tokenizer)
    trainer.train()

    # compute perplexity on the validation set
    val_perplexity = perplexity(model, dataset['val'], tokenizer, batch_size=512)
    print(f'Validation Perplexity: {val_perplexity:.4f}')

    # next word prediction
    model.eval()
    prompt = ["Once upon a time in America, there is a fella called",
                "Once upon a time in the west, there is a cowboy called",
                "Gothenburg is a city in",
                "Who are you?",
                "The capital of China is"]

    for p in prompt:
        input_ids = tokenizer([p], return_tensors='pt', padding=True, truncation=True)['input_ids'].to(model.device)
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            next_token = tokenizer.decode([next_token_id])
            print(f'Prompt: "{p}" -> Next token: "{next_token}"')
