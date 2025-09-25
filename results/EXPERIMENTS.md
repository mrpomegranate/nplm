**Dataset**
- I used "Brown" text corpus. This was a small enough dataset that I can use to train and debug fast for my code. This corpus also has a diverse set of texts from different genres.

**Pre-process**
- I have chunked the corpus into sentence level unit.
- I chose not to remove any punctuations since punctuations are important part of sentence structure. I have only removed special characters that was given in Assignment.md.
- Then I have used nltk word-tokenizer to tokenize each tokens in a sentence.
This tokenizer is simple and easy to use.

**Model**
- I kept the model to one single layer with 256 hidden nodes.
- I noticed that learning rate of 1e-3 casues divergence in train and validation losses.
- I increased the learning rate to 1e-6 and the divergence went away.
- artifacts/training_plot.png shows the results



