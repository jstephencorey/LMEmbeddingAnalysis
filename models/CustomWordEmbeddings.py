from sentence_transformers.util import import_from_string, fullname, http_get
from transformers import PreTrainedTokenizerBase
from typing import List
import torch.nn as nn
import numpy as np
import torch
import json
import os

from utils.constants import MAX_EMBEDDING_SIZE, RANDOM_SEED

# Edited version of the model from MTEB
class CustomWordEmbeddings(nn.Module):
    def __init__(self, 
                 tokenizer: PreTrainedTokenizerBase, 
                 embedding_weights, 
                 update_embeddings: bool = False,
                 embedding_type: str = "full",
                 max_seq_length: int = 1000000):
        nn.Module.__init__(self)
        
        self.embedding_type = embedding_type
        self.update_embeddings = update_embeddings
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.init_embeddings(embedding_weights)
        
    def init_embeddings(self, embedding_weights):
        if isinstance(embedding_weights, list):
            embedding_weights = np.asarray(embedding_weights)

        if isinstance(embedding_weights, np.ndarray):
            embedding_weights = torch.from_numpy(embedding_weights)

        embedding_weights = embedding_weights.to(dtype=torch.bfloat16)
        num_embeddings, embeddings_dimension = embedding_weights.size()
        self.embeddings_dimension = embeddings_dimension
        print(f"Initializing model weights for {self.embeddings_dimension} using {self.embedding_type}")

        if "full" in self.embedding_type:
            self.emb_layer = nn.Embedding(num_embeddings, embeddings_dimension, dtype=torch.bfloat16)
            self.emb_layer.load_state_dict({'weight': embedding_weights})
        elif "random_xn_tok" in self.embedding_type:
            temp_random_array = self.generate_xavier_weights((num_embeddings, MAX_EMBEDDING_SIZE))
            embedding_weights = self.create_tied_weights(temp_random_array, embeddings_dimension)
            self.emb_layer = nn.Embedding(num_embeddings, embeddings_dimension, dtype=torch.bfloat16)
            self.emb_layer.load_state_dict({'weight': embedding_weights})
        elif "random_xn" in self.embedding_type:
            temp_random_array = self.generate_xavier_weights((num_embeddings, MAX_EMBEDDING_SIZE))
            embedding_weights = self.create_tied_weights(temp_random_array, embeddings_dimension)
            self.emb_layer = nn.Embedding(num_embeddings, embeddings_dimension, dtype=torch.bfloat16)
            self.emb_layer.load_state_dict({'weight': embedding_weights})
        elif "crop" in self.embedding_type:
            embedding_size = int(self.embedding_type.split('_')[-1])
            self.emb_layer = nn.Embedding(num_embeddings, embedding_size, dtype=torch.bfloat16)
            self.emb_layer.load_state_dict({'weight': embedding_weights[:,:embedding_size]})
        elif "pad" in self.embedding_type:
            embedding_size = int(self.embedding_type.split('_')[-1])
            temp_random_array = self.generate_xavier_weights((num_embeddings, embedding_size)).to('cuda')
            self.emb_layer = nn.Embedding(num_embeddings, embedding_size, dtype=torch.bfloat16)
            embedding_weights = torch.cat((embedding_weights, temp_random_array[:,embeddings_dimension:]), dim=1)
            self.emb_layer.load_state_dict({'weight': embedding_weights})
            print("PADDING WEIGHTS, WITH SIZE OF ", embedding_weights.size())
        self.emb_layer.weight.requires_grad = self.update_embeddings
        print("Initialized model weights")

    def generate_xavier_weights(self, dimensions):
        torch.manual_seed(RANDOM_SEED)
        weights = torch.empty(dimensions, dtype=torch.bfloat16)
        nn.init.xavier_normal_(weights)
        return weights

    def create_tied_weights(self, base_weights, target_dimensions):
        return base_weights[:, :target_dimensions]
        
    def forward(self, features):
        token_embeddings = self.emb_layer(features['input_ids'])
        cls_tokens = None
        features.update({'token_embeddings': token_embeddings, 'cls_token_embeddings': cls_tokens, 'attention_mask': features['attention_mask']})
        return features

    def tokenize(self, texts: List[str]):
        tokenized_texts = [self.tokenizer.encode(text, max_length=self.max_seq_length) for text in texts]
        # print("Tokenized texts", tokenized_texts)
        sentence_lengths = [len(tokens) for tokens in tokenized_texts]
        # print("sentance lengths", sentence_lengths)
        max_len = max(sentence_lengths)

        input_ids = []
        attention_masks = []
        for tokens in tokenized_texts:
            padding = [0] * (max_len - len(tokens))
            input_ids.append(tokens + padding)
            attention_masks.append([1]*len(tokens) + padding)

        output = {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
                'sentence_lengths': torch.tensor(sentence_lengths, dtype=torch.long)}

        return output



    def get_word_embedding_dimension(self) -> int:
        return self.embeddings_dimension

    def save(self, output_path: str):
        with open(os.path.join(output_path, 'wordembedding_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))
        self.tokenizer.save_pretrained(output_path)
        # self.tokenizer.save(output_path)

    def get_config_dict(self):
        return {'tokenizer_class': fullname(self.tokenizer), 'update_embeddings': self.update_embeddings, 'max_seq_length': self.max_seq_length}

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'wordembedding_config.json'), 'r') as fIn:
            config = json.load(fIn)

        tokenizer_class = import_from_string(config['tokenizer_class'])
        tokenizer = tokenizer_class.from_pretrained(input_path)
        weights = torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu'))
        embedding_weights = weights['emb_layer.weight']
        model = CustomWordEmbeddings(tokenizer=tokenizer, embedding_weights=embedding_weights, update_embeddings=config['update_embeddings'])
        return model