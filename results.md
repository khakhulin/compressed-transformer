## Results

This file contains primary result of compressed model on wmt16.


Num of the epoch - 20, batch size - 150
small-transformer model has 3 block-layers instead of 6


| method\BLEU    | test  | ranks   | 
|----------------|-------|---------|
| original model | 0.440 |  -      | 
| tt-transformer | 0.421 | 2x4x4x2 |
| small-transformer | 0.403 | - | 
| tt-small-transformer | 0.396 | - | 
 

We use tt-decomposition for every fc layer in encoder and decoder  in the following way:
- ranks of the first layer   : 2x4x4x2
- ranks of  the second layer : 4x4x4x4

(ones-dimensions have been omitted)

