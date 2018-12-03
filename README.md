## Requirments

* Install spacy

```
pip install -U spacy
python -m spacy download en
python -m spacy download de
```

* Install torchtext

```
pip install torchtext
```

## Run NMT

* Run (with cuda, if it's available):

```
PYTHONPATH="." python nmt/train.py --seed 45 --lower --min_freq 10
```