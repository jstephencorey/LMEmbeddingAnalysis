# How do the Embedding Layers of LLMs change with scale?

By Joseph Corey

todo:

  - > cut a lot of the fat

  - > See if simple enough for dad to understand most of it?

  - > Post again to Discord to get Stella's thoughts/approval

## Introduction

LLMs have the curious distinction of being near the top of two seemingly
opposing lists, “Most Complex Human Creations”, and ”Least Well
Understood Human Creations”. Despite having architectures designed by
humans, being trained on primarily human output (read: the internet),
and having every bit (literally) exposed to human view, LLMs are still
effectively black boxes when it comes to understanding why they do what
they do.

Digging into understanding how LLMs and other neural networks work
(broadly called “interpretability”) is a key field in Machine Learning
research right now.

This analysis aims to address one of the less well understood pieces of
LLMs, the embedding layer. Specifically how the token embedding layers
in LLMs change as model size and embedding size increases.

The main conclusions from this analysis are that embedding quality
increases with size to a point, but then stagnates or decreases in
quality after a certain model size (though this size differs by model
suite). This is interesting because it implies either an
underutilization/undertraining of embeddings in large models, or a
diminishing importance of embeddings as a model size increases.

The code used for this analysis can be found at
[<span class="underline">https://github.com/jstephencorey/LMEmbeddingAnalysis</span>](https://github.com/jstephencorey/LMEmbeddingAnalysis).

## Embedding Layers in Language Models

This blog post aims to answer the question “As LLM models get larger,
how does the level of information in their embedding layers change?”

As the foundational layer of the model (after the tokenizer), the
embedding layer's quality seems a vitally important step in the language
processing of the model. It transforms text tokens per a predefined
vocabulary, into a fixed-size, meaningful vector in the embedding space
for the neural network's processing, hopefully capturing the tokens'
semantic and syntactic properties for later processing by the model

However, as the embedding layer scales with the overall model size, its
role and optimization become more complex, as I find in this analysis.

Intuition suggests that bigger and more capable models would naturally
harbor more sophisticated embeddings. After all, larger models
consistently outperform their smaller counterparts, a fact that should
logically extend to their token embeddings.

In a larger model, each embedding has more room to hold meaning as a
result of more embedding dimensions, and the model has the capability to
capture and use more nuance in all of its weights. However, this isn’t
entirely consistent with what is found in this analysis. Not all of the
meaning and capability growth in the model as a whole seems to be
captured in the embedding layer of the model.

## Methodology and Analysis

This analysis looks at 5 model suites: Pythia (14m-12b)
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
these models ranges from 128 to 14336 dimensions, and with total model
sizings from 14m to 176b parameters. T5 is added for an example of
embeddings in a non decoder-only architecture.

It is hard to perfectly disentangle embedding size from model size,
given that little effort has gone into making a tiny model with a huge
embedding size, or a huge model with a tiny embedding size. By isolating
some factors (cropping and padding embeddings), evaluating random
baselines, and looking particularly at pythia-1b and pythia-1.4b, the
effects of various variables on embedding quality can begin to be
understood.

All the code for this analysis can be found in
[<span class="underline">this github
repo</span>](https://github.com/jstephencorey/LMEmbeddingAnalysis). A
short summary of how I analyzed the model suites:

1.  > Take the token embedding layer from each model in the suite (Note
    > that this is NOT doing any computation in the model, only taking
    > the embedding lookup table of the model and evaluating its
    > output).

2.  > Use the SentenceTransformer library to combine the model suite’s
    > tokenizer, the embedding layer, and a mean pooling module, to
    > result in a single embedding vector per sentence.

3.  > The model encodes sentences as a single, *d\_model* sized
    > embedding. These embeddings are then evaluated on the SCIDOCS
    > retrieval benchmark, using
    > [<span class="underline">MTEB</span>](https://github.com/embeddings-benchmark/mteb)
    > ([<span class="underline">beir</span>](https://github.com/beir-cellar/beir/tree/main)
    > under the hood). Quality of embeddings is then measured by using
    > [<span class="underline">Normalized Discounted Cumulative Gain
    > (ncgd)</span>](https://en.m.wikipedia.org/wiki/Discounted_cumulative_gain).

> The main thing to know is that **higher is better**. None of these
> models are going to be competitive in embedding rankings, that’s not
> the point. (the highest model as of this writing on the MTEB
> leaderboard for the SCIDOCS retrieval,
> [<span class="underline">all-mpnet-base-v2</span>](https://huggingface.co/sentence-transformers/all-mpnet-base-v2),
> scores a 23.76, whereas the highest model in my analysis scored less
> than 8). The goal is to compare model embedding layers with others in
> the same suite and others in different suites using a common metric.

### Pythia

Pythia is one of the more interesting suites because of how open it is,
including a release of many checkpoints throughout training. Below is a
graph of the following ablations:

![](media/image3.png)

1.  > **Original Embedding Weights**: Take the embeddings as they are
    > from the models, unchanged and just how the first layer of the LM
    > would see them.

The main thing that surprised me when looking at this is that the line
doesn’t just keep going up. If you had to pick one embedding layer of a
model to embed your sentences, you should pick pythia-1b, not, as you
might expect, pythia-12b. This is true in spite of the fact that
pythia-12b is a more capable language model and has a embedding size
twice that of pythia-1b.

Especially of note is pythia-1.4b and pythia-1b. These two models are
the closest of any in this analysis to being a controlled experiment.
These two models have the same embedding size and were initialized with
identical embeddings before pre-training. Pythia-1.4b has 2x as many
attention heads (8 vs. 16) and 1.5x as many hidden layers (16 vs. 24).
Notably, dispute being the larger models, pythia-1.4b scores slightly
*worse*. This seems to imply that, past a given point, making the model
larger makes the embedding layer worse, all else being equal. This is
confirmed in other model suites as well, though the cutoff is not always
the same embedding/model size.

2.  > **Random Embedding Weight**s: Re-initialize the embeddings of each
    > model to a set of random numbers (xavier\_normal). (Note, this is
    > a shared seed, so every dimension for a given token id will be
    > identical across models and suites)

At least some of the improvement as embedding size increases comes
purely from size, as noted by the gentle increase in the random
baseline. The random baseline is further discussed in a later section.

3.  > **Embeddings Cropped to 128 Dims**: Cut down the embeddings of all
    > of the models so they only encode tokens as the first 128
    > dimensions, as a way to measure how much “information per
    > dimension” a given embedding has. (Note that this is identical to
    > the original embedding weights for pythia-14m)

The “information per dimension”, as measured by quality of the first 128
dimensions of each model also seems to peak (if one model size earlier).
This seems to imply that even at the level of the embedding vectors, the
Pythia models smaller than pythia-410m don’t learn as much as they
potentially have the embedding space to.

Also of note is that the first 128 dimensions of the embedding space of
pythia-12b are only marginally better than that of pythia-14m. This
implies that while pythia-12b as a model may be significantly more
capable than pythia-14m, on a per-dimension basis, the embedding vectors
are about as capable/information dense, at least I'm respect to
usefulness for retrieval. Pythia-12b is more capable as a model in part
because of a larger dimensionality, not because it uses its dimensions
better (at least, for the embedding layer).

4.  > **Embeddings Padded to 5012 Dims**: Pad the embeddings with random
    > numbers so each models encodes each token as the model embedding
    > concatenated with a random vector so they are all the same size
    > (5012 dimensions). (Note that this is identical to the original
    > embedding weights for pythia-12b)

Padding the original weights with random numbers helps up to about 512
original dimensions. The trained information in the 128/256/512
dimensions from the model improve notably when noise is added to the
vector (likely because there’s more dimensions to compare with for the
retrieval). This seems to imply that 128 dimensions just isn’t enough to
really capture the full meanings of tokens.

### OPT

The OPT models, released by AI at Meta, are another nice set of models
to look at because they cover a very wide variety of model sizes.

The padding in this model is up to 12288 dimensions, the size of the
embeddings on the largest OPT model, opt-175b. This doesn’t affect the
embeddings much, though, besides the smallest model.

![](media/image1.png)

There are three main things to note here in how this analysis differs
from the Pythia analysis.

First of all, the model improves up to 13b parameters/5120 dimensions
before basically plateauing, as opposed to Pythia’s increase up to 1b
parameters/2048 dimensions and plateauing at that size (technically it's
more “inconsistent” than strictly “plateauing”, but the main point is
that it doesn't improve much at all). It’s unclear why both model suites
plateau, but at very different sizes.

Secondly, the random baseline is better than the smaller models. Unlike
the Pythia suite, where each model outperforms the random baseline,
however the smaller OPT models are trained, they seem to lose
information that would be helpful in encoding information for retrieval.
This could be a initialization issue, or potentially an issue with this
particular metric.

The final thing of note is that retrieval with the first 128 dimensions
of opt-13b or opt-175b actually outperforms retrieval with all 2048
dimensions of opt-1.3b, though all of opt's 128 dimensional chunks
underperform the best of pythia's chunks.

### Other Model Suites

The further individual analysis of the other suites considered in this
blog post -
[<span class="underline">Cerebras</span>](https://github.com/jstephencorey/LMEmbeddingAnalysis/blob/main/plots/Cerebras_MTEB%20SCIDOCS_ndcg_at_10.png),
[<span class="underline">Bloom</span>](https://github.com/jstephencorey/LMEmbeddingAnalysis/blob/main/plots/Bloom_MTEB%20SCIDOCS_ndcg_at_10.png),
and
[<span class="underline">T5</span>](https://github.com/jstephencorey/LMEmbeddingAnalysis/blob/main/plots/T5-v1_1_MTEB%20SCIDOCS_ndcg_at_10.png)
- are all found on the github repo associated with this analysis, as
well as the [<span class="underline">raw
data</span>](https://github.com/jstephencorey/LMEmbeddingAnalysis/tree/main/data)
that went into the graphs in this analysis.

### All Model Suites

Considered next to each other, the models form an interesting
comparison.

### ![](media/image2.png)

This combined analysis shows how, despite some differences, there is a
notable pattern that, past a certain point, embedding quality stagnates
or regresses.

The one exception to this is Cerebras, though given that suite’s score
similarity to OPT, I hypothesize it would plateau soon if the suite kept
increasing. This peak/stagnation point occurs at different sizes in
different model suites. Namely, 1024 dimensions for T5, 2048 dimensions
for Pythia and Bloom, and 5120 dimensions for OPT.

Also notable is that the embeddings are very different quality by suite.
OPT-66b’s embeddings (9216 dimensions) are slightly worse at embedding
than pythia-70m’s embedding (512 dimensions).

### Random baseline from the tokenizers:

![](media/image4.png)

Of course, model suites vary in many ways besides just model size, and
one big change with a potentially large impact is tokenizer choice. I
think tokenizers are understudied, and point to efforts like
[<span class="underline">tokenmonster</span>](https://github.com/alasdairforsythe/tokenmonster)
to rethink the way we consider tokenizing text for LLMs.

Looking at how the tokenization affects the random baselines is rather
interesting. For instance, the fact that Bloom notably improves over the
other models makes sense when you consider that it has a vocab size
almost 5x that of the next largest tokenizer (larger vocabularies make
it easier to differentiate between random tokens). Of most interest to
me, however, is that Cerebras, OPT, and Pythia have almost identical
vocab sizes, but score somewhat differently. (I ran this for a few other
seeds, and though the exact lines slightly vary, the graph overall looks
the same, see the
[<span class="underline">plots</span>](https://github.com/jstephencorey/LMEmbeddingAnalysis/tree/main/plots)
folder for those graphs).

Overall it seems like tokenizers may have some influence on embedding
quality, though nothing to substantially effect this work’s conclusions.

## Implications and Future Work

One notable implication is that past a certain size, embedding quality
seems to go down or stagnate, at least as measured by a retrieval
benchmark.

This implies one of the following (though one does not exclude the
others):

1.  > An underutilization/undertraining of the embedding space in large
    > models. This means that if the models were trained in a way that
    > they had better embedding spaces, the models as a whole would be
    > more capable models.  
    >   
    > This holds a lot of potential for increased model performance if
    > true. Word embeddings predate modern large language models (e.g.
    > [<span class="underline">word2vec</span>](https://arxiv.org/abs/1301.3781)),
    > and applying some effort to things like pre-pre-training the
    > embeddings before or instead of pre-training it in sync with the
    > model, or improving training through efforts like
    > [<span class="underline">μP</span>](https://arxiv.org/abs/2304.06875).

2.  > A diminishing importance of embedding space in large models. By
    > this theory, large models are capable not “in spite of” their
    > comparatively lackluster embeddings, but rather that beyond a
    > certain point, the embeddings don’t play a very significant role
    > in language processing. They act perhaps more as an extension of
    > the token ids, a way to differentiate one token from another, and
    > meaning is encoded more in the processing of further layers.  
    >   
    > This is important because it would make interpretability harder.
    > If, in fact, much of the meaning in tokens is contained in later
    > layers than the token embedding layer, it makes it more difficult
    > to interpret and understand the language models.

3.  > Retrieval is a poor metric for large embedding layer sizes of
    > large language models. Further experiments on these model suites
    > with different metrics is encouraged.

There are several possible explanations for the general downhill trend
or stagnation of models after a certain point:

1.  The models aren’t as fully trained as they could be, and pretraining
    on further data would improve the dimensions as well. This is at
    least mildly supported by [<span class="underline">the original
    Pythia paper</span>](https://arxiv.org/abs/2304.01373), which noted
    that the models (as a whole) still were improving at the end of
    training, and likely would have improved more with further
    training.  
      
    Some evidence against that claim is that that the embeddings usually
    train early on in the model, then stay relatively stable for the
    rest of pretraining (See [<span class="underline">this informative
    blog
    post</span>](https://www.lesswrong.com/posts/2JJtxitp6nqu6ffak/basic-facts-about-language-models-during-training-1#Tokenizer_Embeddings_are_rapidly_learnt_then_stabilize).
    My personal analysis of the Pythia models found that a large amount
    of the changes in the embedding layers of a model happen in the
    first \~10% of the training, and after that point, the changes to
    the embedding layer slow down significantly.) If the embedding layer
    is relatively set early on, and doesn’t change significantly in the
    later parts of pretraining, then further training wouldn’t make a
    significant difference. This is an open area for further research.

2.  A similar possibility is that the hyperparameters (e.g. learning
    rate) for pretraining the layers of large language model aren’t
    necessarily the best hyperparameters for pretraining the embedding
    layer. This is proposed in the
    [<span class="underline">μP</span>](https://arxiv.org/abs/2304.06875)
    paper, for instance.  
      
    However, the real life performance the Cerebras model (which was
    trained with μP hyperparameter recommendations) would tend to
    suggest that that alone is not enough.

3.  The bigger the model, the less important the embedding layer. Models
    like
    [<span class="underline">EELBERT</span>](https://arxiv.org/abs/2310.20144)
    have shown that capable models can be trained even if the embedding
    layer is more or less a hash function (in their paper, an “n-gram
    pooling hash function”), so it seems like as long as there is some
    information in the embeddings to distinguish between tokens, the
    rest of the language model is capable of transforming that
    information (pun intended) into useful next-word predictions. As the
    model gets bigger, it can use embeddings that are as good as or
    worse than smaller models’ embedding layers to get comparatively
    better results because of all of the other information stored in
    other weights in the model.  
      
    For future work, it may be useful to train a model while freezing
    the random embedding weights at initialization, and compare that
    training to a model with unfrozen weights. This could help answer
    how important capable trained embeddings are in a language model.
    Especially in small models, the embeddings takes up a significant
    percentage of the weights, so something like the hashing proposed in
    EELBERT could be worth the tradeoffs in time and (potentially)
    quality.

4.  It’s also possible that retrieval isn’t the best metric for
    measuring capabilities in embedding layers in models, and/or that
    larger models are learning things, just not the things measured by
    the metric picked for evaluation in this blog post. Repeating these
    evaluations using other metrics is an open task for future work.

## Conclusions

The embedding layer of various LLM models gets more capable (as measured
by ability to be used as embeddings in a retrieval benchmark) with size
up to a point. This may point to the inability of small models to have
enough embedding space to store all the information they could need, and
the potential underuse of embedding dimensions by larger models.

This requires more analysis in the future. Most especially, it may be
worth it to explore if embeddings are underutilized, or merely less
important in larger models. This could hold implications for how to
pre-train large language models

Whatever future work shows regarding the importance of embeddings in
large models, this work can stand as evidence that bigger is not always
better in every aspect of LLMs, and hopefully leads to a greater
understanding of how LLMs work, scale, and learn.
