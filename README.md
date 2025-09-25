# A Neural Probabilistic Language Model 
Bengio et al. introduced the concept of embedding each word in a feature vector to overcome the issue of semantics and very large sparse matrix computation. This feature vector projects words into a higher dimension. The weights of the vectors are learned and as a result, words similar in nature, semantics and context lie closer to each other in higher dimension  

In order to run this code, follow the steps:

- clone the repo 
```
https://github.com/mrpomegranate/nplm.git
```
- cd to the root directory ./nplm
- execute the train.py
```
python -m nplm.train --n 5 --device cuda --epochs 50 --max_vocab 40000 --lr .000001 --batch_size 512 --vocab artifacts/brown_word_vocab.json
```
- Once the exection is over, predict on the text as follows (example n = 5. context_size is 4 in order to predict the 5th word)
```
python -m nplm.predict --context "Bike is good looking"
```
