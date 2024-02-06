from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer
import torch
import os
import gc

from models.CustomWordEmbeddings import CustomWordEmbeddings
from utils.constants import HOME_DIR

def get_saved_embedding(embedding_filename, device='cpu'):
    full_embedding = torch.load(embedding_filename, map_location=torch.device(device))
    weights = full_embedding.weight.data

    embedding = torch.nn.Embedding.from_pretrained(weights)
    embedding.requires_grad_(False) # Freeze or unfreeze)
    return embedding

def make_dir_if_none(dirname):
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

def clear_cuda_memory(obj):
    del obj
    obj = None
    gc.collect()
    torch.cuda.empty_cache()
    with torch.no_grad():
        torch.cuda.empty_cache()

def get_model(name,
              embedding_type: str="full",
              pooling_mode: str="mean",
              tokenizer: str='bigscience/tokenizer',
              embedding_size: tuple=None,
              save_model: bool=False):
    # tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    # tokenizer = GPT2TokenizerFast.from_pretrained("facebook/opt-125m")
    if embedding_size is None:
        embedding_weight = get_saved_embedding(embedding_filename=f"../embeddings/{name}.pth").weight
    else:
        vocab_side = embedding_size[0]
        d_model = embedding_size[1]
        # print(vocab_side,d_model)
        embedding_layer = torch.nn.Embedding(vocab_side, d_model)
        embedding_weight = embedding_layer.weight

    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    word_embedding_model = CustomWordEmbeddings(tokenizer, 
                                                embedding_weight,
                                                embedding_type=embedding_type,
                                                max_seq_length=2048)
    output_folder = f"{HOME_DIR}/mteb_analyses/{name}_{embedding_type}_{pooling_mode}/"
    make_dir_if_none(output_folder)
    if save_model:
        model_folder = f"{output_folder}model/"
        make_dir_if_none(model_folder)
        word_embedding_model.save(model_folder)
        word_embedding_model = CustomWordEmbeddings.load(model_folder)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    return model, output_folder