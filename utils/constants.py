MAX_EMBEDDING_SIZE=14336 # needs to match the largest embedding model you have, at least
BATCH_SIZE = 16 # needed to not go OOM with MTEB. Can be 8 or 16 for my 4090, haven't tested on other systems.
HOME_DIR="/mnt/f/Dev/LMEmbeddingAnalysis" #change as necessary
HOME_DIR_WIN="F:/Dev/LMEmbeddingAnalysis" #change as necessary
RANDOM_SEED=42