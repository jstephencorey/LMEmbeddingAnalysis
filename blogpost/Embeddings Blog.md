# How do the Embedding Layers of LLMs change with scale?

By Joseph Corey

## Introduction

LLMs are one of the most complex things humans have built while also
being one of the least well understood things humans have built. Despite
several interpretability fields springing up around machine learning
models (e.g. mechanistic interpretability), there are still lots of
things that remain imperfectly understood or entirely unexplained. This
blog aims to address one less well understood aspects of LLMs, the
embedding layer. Specifically how the token embedding layers in LLMs
change as model size and embedding size increases.

The main conclusions from this analysis are that embedding quality
increases with size to a point, but then stagnates or decreases in
quality after a certain model size. This is interesting because it
implies either an underutilization of embeddings in large models, or a
diminishing importance of embeddings as a model size increases.

The code used for this blog can be found at
[<span class="underline">https://github.com/jstephencorey/LMEmbeddingAnalysis</span>](https://github.com/jstephencorey/LMEmbeddingAnalysis).

## Analysis

This blog aims to examine how quality of token embeddings scale as
embedding and model size both increase. Specifically, it aims to answer
the question “As LLM models get larger, how does the level of
information in their embedding layers change?”

This analysis looks at 5 model suites, with total model sizings from 14m
to 176b parameters. Pythia (14m-12b)
([<span class="underline">Huggingface</span>](https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1),
[<span class="underline">Paper</span>](https://arxiv.org/abs/2304.01373)),
OPT (125m-175b)
([<span class="underline">Huggingface</span>](https://huggingface.co/facebook/opt-125m),
[<span class="underline">Paper</span>](https://arxiv.org/abs/2205.01068)),
Cerebras (111m-13b)
([<span class="underline">Huggingface</span>](https://huggingface.co/cerebras),
[<span class="underline">Paper</span>](https://arxiv.org/abs/2304.03208)),
BLOOM (560m-176b)
([<span class="underline">Huggingface</span>](https://huggingface.co/bigscience/bloom),
[<span class="underline">Paper</span>](https://arxiv.org/abs/2211.05100)),
and Google’s T5 v1.1 (60m-11b)
([<span class="underline">Huggingface</span>](https://huggingface.co/collections/google/t5-release-65005e7c520f8d7b4d037918),
[<span class="underline">Paper</span>](https://arxiv.org/abs/1910.10683)).
These aren't necessarily the most modern or powerful models, but they
are suites with large ranges of sizes, which can be used for examining
scaling on embedding quality. The embedding dimensionality sizes of
these models ranges from 128 to 14336 dimensions.

It is hard to perfectly disentangle embedding size from model size,
given that little effort has gone into making a tiny model with a huge
embedding size, or a huge model with a tiny embedding size. By isolating
some factors (cropping and padding embeddings), evaluating random
baselines, and looking particularly at two models with identical
embedding sizes, embedding initialization, and training corpus
(pythia-1b and pythia-1.4b), the effects of various variables on
embedding quality can be understood.

## Methodology

All the code for this analysis can be found in
[<span class="underline">this github
repo</span>](https://github.com/jstephencorey/LMEmbeddingAnalysis). A
short summary is this:

For each model suite:

1.  > Take the token embedding layer from each model in the suite (Note
    > that this is NOT doing any computation in the model, only taking
    > the embedding lookup table of the model and evaluating its
    > output). I change this slightly across three of the following four
    > tests:
    
    1.  > Take the embeddings as they are from the models, just how the
        > first layer of the LM would see them
    
    2.  > Re-initialize the embeddings of each model to a set of random
        > numbers (xavier\_normal). (Note, this is a shared seed, so
        > every dimension for a given token id will be identical across
        > models and
    
    3.  > Cut down the embeddings of all of the models so they only
        > encode tokens as the first 128 dimensions, as a way to measure
        > how much “information per dimension”
    
    4.  > Pad the embeddings with random numbers so each models encodes
        > each token as an 14336 dimensional vector.

2.  > Using the SentanceTransformer library to combine the model suite’s
    > tokenizer, the embedding layer, and a mean pooling module, to
    > result in a single embedding vector per sentence.

3.  > Quality of embeddings is then measured by using mean average
    > precision (map) from the SCIDOCS retrieval benchmark, evaluated
    > using
    > [<span class="underline">MTEB</span>](https://github.com/embeddings-benchmark/mteb)
    > (under the hood it uses
    > [<span class="underline">beir</span>](https://github.com/beir-cellar/beir/tree/main)).
    > A higher number means that set of embeddings did better.

## Analysis by Model And Embedding size

### Pythia

Pythia is one of the more interesting suites because of how open it is,
including a release of many checkpoints throughout training. One
particularly interesting finding as a result of these released
checkpoints is that the embeddings usually train early on in the model,
then stay relatively stable for the rest of pretraining (See
[<span class="underline">this informative blog
post</span>](https://www.lesswrong.com/posts/2JJtxitp6nqu6ffak/basic-facts-about-language-models-during-training-1#Tokenizer_Embeddings_are_rapidly_learnt_then_stabilize).
My personal analysis has found that even as early as \~10% of the way
through training, the embedding layer is very similar to it’s final
trained state.)

![](media/image7.png)

The original weights tend to improve as size increases until about
pythia-1b, then the embeddings do slightly worse as the size increases.

Especially of note is pythia-1.4b and pythia-1b. These two models are
the closest of any in this analysis to being a controlled experiment.
These two models have the same embedding size, but pythia-1.4b has 2x as
many attention heads (8 vs. 16) and 1.5x as many hidden layers (16 vs.
24). Both pythia-1b and pythia-1.4b were initialized with identical
embeddings before pre-training. This seems to imply that, past some
point, making the model larger makes the embedding layer worse, all else
being equal.

At least some of the improvement as embedding size increases comes
purely from size, as noted by the gentle increase in the random
baseline.

Additionally, padding the original weights with random numbers helps up
to about 512 dimensions. The trained information in the 128/256/512
dimensions from the model, combined with improvements in score from the
quantity of random dimensions (see how much more capable the random 5120
result is than the trained 128 dimensional model, for instance) leads to
a result higher than the random or trained vectors on their own.

The “information per dimension”, as measured by quality of the first 128
dimensions of each model, seems to peak around 768 dimensions, or
pythia-160m. Note that the first 128 dimensions of pythia-410m are
better for retrieval than all 128 dimensions of pythia-14m. This seems
to imply that even at the level of the embedding vectors, the Pythia
models smaller than pythia-410m don’t learn as much as they potentially
have the space to.

Also of note is that the first 128 dimensions of the embedding space of
pythia-12b are only marginally better than that of pythia-14m. This
implies that while pythia-12b as a model may be significantly more
capable than pythia-14m, on a per-dimension basis, the embedding vectors
are about as capable. Pythia-12b is more capable as a model in part
because of a larger dimensionality, not that it uses it’s dimensions
better (at least, for the embedding layer)

### OPT

![](media/image6.png)

Things to note:

  - > Vaguely positive the whole time

  - > Only really increases up to \~13b, OPT-175b is only barely better
    > (and slightly worse with just the first 128 dims)

  - > Peak here, 13b/5120 dims, is different from pythia (1b/2048 dims)
    > and bloom (1.7b/2048 dims)

### Cerebras

![](media/image3.png)

Things to note:

  - > Vaguely increasing all the way

  - > Not increasing in first 128 dim evaluation

  - > Worse than random for most of it

  - > Note about μP

### BLOOM

![](media/image2.png)

Things to note:

  - > Worse than random in small models

  - > Peaks at 1b7/2048 dims, and stays constant from there

  - > First 128 dims gets worse after 3b/2560 dims

## T5

### ![](media/image5.png)

Things to note: NOTE TODO, this is wrong, need to change stuff with the
padding

  - > This pattern isn’t as pronounced (less range), but still seems to
    > be there

  - > Not just an “decoder only” problem

  - > Paper: [<span class="underline">1910.10683.pdf
    > (arxiv.org)</span>](https://arxiv.org/pdf/1910.10683.pdf)

### All Model Suites

### ![](media/image4.png)

This combined analysis shows that the various model suites have somewhat
different characteristics, even if they have similar patterns. Notably,
the various suites improve embedding quality with size up to a point,
then plateau or decrease in score (with the exception of Cerebras).
However, this peak occurs at different sizes in different model suites.
Notably, 2048 dimensions for Pythia and Bloom, and 5120 dimensions for
OPT.

It would be reasonable to expect that larger models should have more
capable embeddings. The model as a whole is more capable, each embedding
has more room to hold meaning as a result of more embedding dimensions,
and the model has the capability to capture more nuance. However, this
isn’t what is shown.

Also notable is that the embeddings are very different quality by suite.
OPT-66b’s embeddings (9216 dimensions) are slightly worse at embedding
than pythia-70m’s embedding (512 dimensions).

There are several possible explanations for the general downhill trend
or stagnation of models after a certain point:

1.  The models aren’t as fully trained as they could be, and pretraining
    on further data would improve the dimensions as well. This is at
    least mildly supported by [<span class="underline">the original
    Pythia paper</span>](https://arxiv.org/abs/2304.01373), which noted
    that the models (as a whole) still were improving at the end of
    training, and likely would have improved more with further
    training.  
      
    Some evidence against that claim would be the idea mentioned above
    that the embedding layer doesn’t change significantly in the later
    parts of pretraining. If the embedding layer is relatively set early
    on, then further training wouldn’t make a significant difference.

2.  The bigger the model, the less important the embedding layer. Models
    like
    [<span class="underline">EELBERT</span>](https://arxiv.org/abs/2310.20144)
    have shown that capable models can be trained even if the embedding
    layer is more or less a hash function, so it seems like as long as
    there is some information in the embeddings to distinguish between
    tokens, the rest of the language model is capable of transforming
    that information (pun intended) into useful next-word predictions.
    As the model gets bigger, it can use comparatively worse embeddings
    to get comparatively better results because of all of the other
    information stored in other weights in the model.  
      
    For future work, it may be useful to train a model while freezing
    the random embedding weights at initialization, and compare that
    training to a model with unfrozen weights. This could help answer
    how important capable trained embeddings are in a language model.
    Especially in small models, the embeddings takes up a significant
    percentage of the weights, so something like the hashing proposed in
    EELBERT could be worth the tradeoffs in time and (potentially)
    quality.

3.  It’s also possible that retrieval isn’t the best metric for
    measuring capabilities in embedding layers in models, and/or that
    larger models are learning things, just not the things measured by
    the metric picked for evaluation in this blog. Repeating these
    evaluations using other metrics is an open task for future work.

## Random baseline from the tokenizers:

![](media/image1.png)

## Implications and Future Work

One notable implication is that past a certain size, embedding quality
seems to go down or stagnate, at least as measured by a retrieval
benchmark.

This implies one of the following (though one does not exclude the
others):

1.  > An underutilization of the embedding space in large models. This
    > means that if the models were trained in a way that they had
    > better embedding spaces, the models as a whole would be more
    > capable models.  
    >   
    > This holds a lot of potential for increased model performance if
    > true. Word embeddings predate language models (e.g.
    > [<span class="underline">word2vec</span>](https://arxiv.org/abs/1301.3781)),

2.  > A diminishing importance of embedding space in large models. By
    > this theory, large models are capable not “in spite of” their
    > comparatively lackluster embeddings, but rather that beyond a
    > certain point, the embeddings don’t play a very significant role
    > in language processing. They act perhaps more as an extension of
    > the token ids, a way to differentiate one token from another, and
    > meaning is encoded more in the processing of further layers.  
    >   
    > This is important because it makes interpretability harder. If, in
    > fact, much of the meaning in tokens is contained in later layers
    > than the token embedding layer, it makes it more difficult to
    > interpret and understand the language models.

3.  > Retrieval is a poor metric for large embedding layer sizes of
    > large language models.

## Conclusions

The embedding layer of various LLM models gets more capable (as measured
by ability to be used as embeddings in a retrieval benchmark) with size
up to a point. This may point to the inability of small models to have
enough embedding space to store all the information they could need, and
the potential underuse of embedding dimensions by larger models.

This requires more analysis in the future. Most especially, it may be
worth it to explore if embeddings are underutilized, or merely less
important in larger models. This could hold implications for how to
pre-train large model, either through pre-pre-training the embedding, or
using a method like
[<span class="underline">μP</span>](https://arxiv.org/abs/2304.06875)
that trains the embedding differently than the rest of the model (with a
different learning weight).

Whatever future work shows regarding the importance of embeddings in
large models, this work can stand as evidence that bigger is not always
better in every aspect of LLMs, and hopefully leads to a greater
understanding of how LLMs work, scale, and learn.
