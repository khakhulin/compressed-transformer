
### Dataset IWSL14
* Run the script (borrowed from [Harvard NLP repo](https://github.com/harvardnlp/BSO/tree/master/data_prep/MT)) to download and preprocess IWSLT'14 dataset:
```shell
$ cd preprocessing
$ source prepareData.sh
```
__warning__: this script requires Lua and luaTorch. As an alternative, you can download all necessary files(data directory) from [this repo](https://github.com/pcyin/pytorch_nmt/tree/master/data) or via this [link](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/pcyin/pytorch_nmt/tree/master/data)

* Generate Vocabulary Files

```
python vocab.py
```
Example:
```
python vocab.py --train_src data/train.de-en.de --train_tgt data/train.de-en.en --output data/vocab.bin
```