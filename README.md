

We use WMT16(Multi30k) dataset. Please use the batch as large as possible. 
Net can converge After 5 epoch with batch size 128.
   
## Requirments

* Install nltk
* Install tntorch
* Install torchtext

```
pip install torchtext
```

## Run NMT

* Run (with cuda, if it's available):

```
PYTHONPATH="." python3 nmt/train.py --seed 45  --save_model_after 1000 \
 --valid_max_num 120  --lower --min_freq 3 --lower --tokenize --batch 82
 ```
 
 
 For train compressed model:

```
PYTHONPATH="." python3 nmt/train.py --seed 45  --save_model_after 1000 \
 --valid_max_num 120  --lower --min_freq 3 --lower --tokenize --batch 82 --compress --exp compressed

 ```
 
 _Note_: use multi-gpu mode 
 
 For test use:
 ```
 sh scripts/run_test.sh $path_to_your_uncompressed_model
 ```
 or for compressed:
  ```
 sh scripts/run_test_compressed.sh $path_to_your_compressed_model
 ```
 