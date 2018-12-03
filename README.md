## Requirments

* Install nltk

* Install torchtext

```
pip install torchtext
```

## Run NMT

* Run (with cuda, if it's available):

```
PYTHONPATH="." python nmt/train.py --seed 45 --lower --tokenize --min_freq 10 
```