

We use WMT16(Multi30k) dataset. Please use the batch as large as possible. 
Net can converge After 5 epoch with batch size 128.
   
## Requirments

* Install nltk

* Install torchtext

```
pip install torchtext
```

## Run NMT

* Run (with cuda, if it's available):

```
PYTHONPATH="." python3 nmt/train.py --seed 45  --save_model_after 2000 \
 --valid_max_num 120  --lower --min_freq 1 --lower --tokenize --batch 128
 ```