* Install spacy

```
pip install -U spacy
python -m spacy download en
python -m spacy download de
```

* Run (with cuda, if it's available):

```
PYTHONPATH="." python nmt/train.py --seed 45
```