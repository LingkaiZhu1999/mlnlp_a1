
import torch, pickle
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from torch.utils.data import DataLoader
import numpy as np
import sys, time, os

# Mine Libraries
import tqdm
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence

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
    # text_bytes = [t.encode('utf-8') for t in text]
    encoded_text = text.encode("utf-8")
    bytes_list = [bytes([b]) for b in encoded_text]
    return bytes_list
    

def merge_pairs(tokens, best_pair):
    # merge tokens based on best pair
    i=0
    merged_tokens = []
    merged_token = b''.join(best_pair)
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
            merged_tokens.append(merged_token)
            i += 2  
        else:
            merged_tokens.append(tokens[i])
            i += 1
    return merged_tokens    

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
        
        Returns:
            Vocabulary object that maps each token to an integer ID.
    """

    # TODO: build the vocabulary, using a byte-level BPE tokenization (explained above)
    # Then return a tokenizer object (implemented below).
    # add <BOS> at the beginning of each paragraph; <EOS> at the end of each paragraph; <PAD> for padding to make input tensors rectangular

    #-- 1. Split every line of the training text into individual UTF-8 bytes.
    with open(train_file, encoding='utf-8') as f:
        train_texts = [line.strip() for line in f if line.strip()]  # each line is a paragraph, remove empty lines and whitespace

    ## Each row will be a list of words in corresponding line
    train_words = []
    for Line_idx in tqdm.trange(0, len(train_texts), 1):
        train_words.append([word for word in train_texts[Line_idx]])

    ## Each will be a list of bytes in corresponding line
    train_words_bytes = []
    
    for r in tqdm.trange(len(train_words)):
        train_words_bytes.append([tokens for word in train_words[r] for tokens in split_into_bytes(word)])
    
    # counts_train_bytes = dict(sorted(Counter(flatten_train_words_bytes).items(), key=lambda x: x[1], reverse=True))
    # vocabulary = {Ntoken: i for i, Ntoken in enumerate(counts_train_bytes.keys())}

    #-- 2. Start the vocabulary with the four special tokens and all 256 possible byte values. The vocabulary maps each token to an integer ID.
    special_tokens = [pad_token, eos_token, bos_token, unk_token]
    byte_tokens = [bytes([i]) for i in range(256)]
    vocabulary = {token: i for i, token in enumerate(special_tokens + byte_tokens)}

    #-- 3. Count how often each pair of neighboring tokens occurs.
    def counter_pairs(train_words_bytes):
        pairs_counter = Counter()
        for line in train_words_bytes:
            pairs_counter.update(zip(line[:-1], line[1:]))
        best_pair, count = pairs_counter.most_common(1)[0]
        return best_pair, count


    #-- 4. Join the most common pair everywhere it occurs. Add the joined token to the vocabulary and save the pair as a merge rule.
    merge_rules = []
    for s in tqdm.trange(max_voc_size - 260):
        best_pair, count = counter_pairs(train_words_bytes)

        train_words_bytes = [merge_pairs(line, best_pair) for line in train_words_bytes]

        vocabulary.update({(b''.join(best_pair)): len(vocabulary)})
        merge_rules.append(best_pair)

        print(f'Current vocabulary size: {len(vocabulary)}, most common pair: {best_pair}, count: {count}')

    return vocabulary, merge_rules


class A1Tokenizer:
    """A minimal implementation of a tokenizer similar to tokenizers in the HuggingFace library."""

    def __init__(self, model_max_length, pad_token_id, vocabulary, merge_rules=None):
        # TODO: store all values you need in order to implement __call__ below.
        self.pad_token_id = pad_token_id     # Compulsory attribute.
        self.model_max_length = model_max_length # Needed for truncation.
        self.vocabulary = vocabulary
        self.merge_rules = merge_rules

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
        encoded_texts = []
        attention_masks = []
        train_words = []
        train_words_bytes = [] 

        for Line_idx in range(0, len(texts), 1):
            train_words.append([word for word in texts[Line_idx]])
        for r in range(len(train_words)):
            train_words_bytes.append([tokens for word in train_words[r] for tokens in split_into_bytes(word)])
        for items in train_words_bytes:
            for pair in self.merge_rules:
                items = merge_pairs(items, pair)

            ## map the resulting tokens to their vocabulary IDs
            items_id = [self.vocabulary['<BOS>']] + [self.vocabulary.get(token, self.vocabulary['<UNK>']) for token in items] + [self.vocabulary['<EOS>']]
            encoded_texts.append(items_id)
            attention_masks.append([1] * len(items_id))  # Set attention mask to 1 for real tokens

        if truncation:
            encoded_texts = [
                ids[:self.model_max_length - 1] + [self.vocabulary['<EOS>']]
                if len(ids) > self.model_max_length else ids
                for ids in encoded_texts
            ]
            attention_masks = [[1] * len(ids) for ids in encoded_texts]

        if padding:
            # Pad the encoded texts to the same length
            max_length = max(len(text) for text in encoded_texts)
            for i in range(len(encoded_texts)):
                while len(encoded_texts[i]) < max_length:
                    encoded_texts[i].append(self.vocabulary['<PAD>'])
                    attention_masks[i].append(0)  # Set attention mask to 0 for padding tokens


        if return_tensors == 'pt':
            encoded_texts = torch.tensor(encoded_texts, dtype=torch.long)
            attention_masks = torch.tensor(attention_masks, dtype=torch.long)

        # TODO: Return a BatchEncoding where input_ids stores the result of the integer encoding.
        # Optionally, if you want to be 100% HuggingFace-compatible, you should also include an 
        # attention mask of the same shape as input_ids. In this mask, padding tokens correspond
        # to the the value 0 and real tokens to the value 1.
        # if return_tensors is None:
            
            
            
        return BatchEncoding({'attention_mask': attention_masks, 'input_ids': encoded_texts})

    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.vocabulary)

    def decode(self, ids):
        """Return a string decoded from given input ids"""
        id_to_token = {idx: token for token, idx in self.vocabulary.items()}
        special_ids = {
            self.vocabulary["<PAD>"],
            self.vocabulary["<BOS>"],
            self.vocabulary["<EOS>"],
            self.vocabulary["<UNK>"],
        }

        token_to_str = [id_to_token[int(idx)] for idx in ids if int(idx) not in special_ids]
        return b"".join(token_to_str).decode("utf-8", errors="replace")
    
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
        # map the token integers to floating point vectors
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size, padding_idx=0)
        # some RNN variant like nn.LSTM or nn.GRU
        self.rnn = nn.GRU(config.embedding_size, config.hidden_size, batch_first=True)
        # computes the logits of a probability distribution over the vocabulary
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

        X = labels[:, :-1] # all columns in input_ids except the last one
        Y = labels[:, 1:] # all columns in input_ids except the first one

        X[X == -100] = self.embedding.padding_idx

        assert(X.shape == Y.shape)

        embedded = self.embedding(X)
        rnn_out, _ = self.rnn(embedded)
        out = self.unembedding(rnn_out)

        #  compute the loss for the model output and Y 
        loss = loss_func(out.reshape(-1, out.size(-1)), Y.reshape(-1))

        return CausalLMOutputWithPast(loss=loss, logits=out)


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


        def collate_fn(batch):
            input_ids = [
                torch.tensor(item["input_ids"], dtype=torch.long)
                for item in batch
            ]

            input_ids = pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=0
            )

            return {
                "input_ids": input_ids
            }


        # TODO: Relevant arguments: args.per_device_train_batch_size, args.per_device_eval_batch_size
        train_loader = DataLoader(self.train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(self.eval_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, collate_fn=collate_fn)
        
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

        for epoch in tqdm.trange(args.num_train_epochs):

            self.model.train()

            for batch in tqdm.tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.num_train_epochs}'):
                labels = torch.tensor(batch['input_ids'], dtype=torch.long, device=device)
                labels[labels == self.processing_class.pad_token_id] = -100  # Replace padding
                loss = self.model(labels)['loss']
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm.tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}/{args.num_train_epochs}'):
                    labels = torch.tensor(batch['input_ids'], dtype=torch.long, device=device)
                    labels[labels == self.processing_class.pad_token_id] = -100  # Replace padding
                    loss = self.model(labels)['loss']
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f'Epoch {epoch + 1}/{args.num_train_epochs}, Training Loss: {loss.item():.4f}, Validation Loss: {avg_val_loss:.4f}')

        print(f'Saving to {args.output_dir}.')
        self.model.save_pretrained(args.output_dir)

    
if __name__ == '__main__':

    os.makedirs('output', exist_ok=True)
    if not os.path.exists('output/vocabulary.pkl'):
        vocabulary, merge_rules = build_tokenizer('train.txt', max_voc_size=500)
        pickle.dump((vocabulary, merge_rules), open('output/vocabulary.pkl', 'wb'))
    else:
        vocabulary, merge_rules = pickle.load(open('output/vocabulary.pkl', 'rb'))
    print('Vocabulary size:', len(vocabulary))

    tokenizer = A1Tokenizer(model_max_length=128, pad_token_id=0, vocabulary=vocabulary, merge_rules=merge_rules)
    # test_texts = ['This is the test.', 'Another test.']
    dataset = load_dataset('text', data_files={'train': 'train.txt', 'val': 'val.txt'})
    dataset = dataset.filter(lambda x: x['text'].strip() != '')  # Remove empty lines

    if not os.path.exists('output/dataset.pkl'):
        dataset = dataset.map(
            lambda batch: dict(tokenizer(batch['text'], truncation=True, padding=True)),
            batched=True,
            remove_columns=['text'],
        )
        pickle.dump(dataset, open('output/dataset.pkl', 'wb'))
    else:
        dataset = pickle.load(open('output/dataset.pkl', 'rb'))

    A1RNNModelConfig = A1RNNModelConfig(vocab_size=len(vocabulary), embedding_size=32, hidden_size=64)
    A1RNNModel = A1RNNModel(A1RNNModelConfig)
    A1Trainer = A1Trainer(model=A1RNNModel, args=type('args', (object,), {
        'optim': 'adamw_torch',
        'eval_strategy': 'epoch',
        'use_cpu': True,
        'learning_rate': 0.001,
        'num_train_epochs': 5,
        'per_device_train_batch_size': 2,
        'per_device_eval_batch_size': 2,
        'output_dir': 'output'
    })(), train_dataset=dataset['train'], eval_dataset=dataset['val'], processing_class=tokenizer)

    # A1Trainer.train()

    model = A1RNNModel.from_pretrained('output')

    model.eval()
    test_texts = ['This is the test.', 'Another test.']
    test_dataset = tokenizer(test_texts, truncation=True, padding=True, return_tensors='pt')
    test_dataset = {k: v for k, v in test_dataset.items() if k in ['input_ids', 'attention_mask']}
    test_dataset = {k: v.to(model.device) for k, v in test_dataset.items()}

    special_ids = [tokenizer.pad_token_id, tokenizer.vocabulary['<EOS>'], tokenizer.vocabulary['<BOS>'], tokenizer.vocabulary['<UNK>']]
    
    with torch.no_grad():
        input_ids = test_dataset['input_ids']
        logits = model(input_ids)['logits'][0, -1]
        logits[special_ids] = float('-inf') # never predict the special
        tokensprobs = torch.softmax(logits, dim=-1)
        top = torch.topk(tokensprobs, k=5)
        next_token = tokenizer.decode([top.indices[0].item()])
        print(f'Prompt: "{test_texts[0]}" -> Next token: "{next_token}"')
        generated = input_ids[0, :-1].tolist()
        for _ in range(100):
            inp = torch.tensor([generated + [tokenizer.vocabulary['<EOS>']]], dtype=torch.long, device=model.device)
            step_logits = model(inp)['logits'][0, -1]
            step_logits[special_ids] = float('-inf')
            generated.append(step_logits.argmax().item())
        print(f' Continuation {tokenizer.decode(generated)}')
        import pdb; pdb.set_trace()

    


    print('done')
