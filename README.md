## Code for "Distantly Supervised Named Entity Recognition via Confidence-Based Multi-Class Positive and Unlabeled Learning" published at ACL 2022
The code is developed based on https://github.com/v-mipeng/LexiconNER.

## Note:
MPU and Conf-MPU with BERT will be released soon. (If it's urgent for you to perform experiments with them, please feel free to email me, and I will send you the code which is not organized very well for now.)

## Environment:
Python 3.7, pytorch 1.4

## Need to be downloaded:
Download glove and bio-embedding and put them in `./data`. See the links in our paper.


## How to run:

### For Conf-MPU:

> #### Example 1: train on BC5CDR_Dict_1.0
> ```
> Step 1: python pu_main.py --type bnPU --dataset BC5CDR_Dict_1.0 --flag Entity --m 10 --determine_entity True --embedding bio-embedding --epochs 100
> ```
> Step 1 is to do a binary classification (i.e., `\lambda`) in token-level to determine entity tokens. The trained model will be saved in
> `./saved_model`.
> 
> ```
> Step 2: python pu_main.py --type bnPU --dataset BC5CDR_Dict_1.0 --add_probs True --flag ALL --added_suffix entity_prob --embedding bio-embedding
> ```
> Step 2 is to append predicted confidence scores to the training data. The generated training data will be the input of the model with Conf-MPU risk
 estimation. You need to manually pass the path of the saved model in step 1 in `pu_main.py` (in the block of `elif args.add_probs:`).
> 
> ```
> Step 3: python pu_main.py --type conf_mPU --dataset BC5CDR_Dict_1.0 --flag ALL --suffix entity_prob --m 28 --eta 0.5 --lr 0.0005 --loss MAE
>  --embedding bio-embedding --epochs 100
> ```
> Step 3 is to train the model with Conf-MPU risk estimation.


> #### Example 2: train on CoNLL2003_Dict_1.0
> ```
> Step 1: python pu_main.py --type bnPU --dataset CoNLL2003_Dict_1.0 --flag Entity --m 15 --determine_entity True --epochs 100
> Step 2: python pu_main.py --type bnPU --dataset CoNLL2003_Dict_1.0 --add_probs True --flag ALL --added_suffix entity_prob
> Step 3: python pu_main.py --type conf_mPU --dataset CoNLL2003_Dict_1.0 --flag ALL --suffix entity_prob --m 15 --eta 0.5 --lr 0.0005 --loss
>  MAE --epochs 100
> ```

##### Note:
1. Training data with probabilities of each token being an entity token have been already put in each dataset folder. You can skip step 1 and 2.
2. Hyper-parameters in commands: `m` is a class weight to put more weight on the risk of positive data; `eta` is the threshold `\tau` in Conf-MPU
 risk formula. You can use the same `m` value for other datasets distantly labeled by various dictionaries (i.e., `m = 28` for BC5CDR datasets, and
  `m = 15` for CoNLL2003 datasets).
  

### For MPU:

> #### Example 1: train on BC5CDR_Dict_1.0
> ```
> python pu_main.py --type mPU --dataset BC5CDR_Dict_1.0 --flag ALL --m 28 --lr 0.0005 --loss MAE --embedding bio-embedding --epochs 100
> ```

> #### Example 2: train on CoNLL2003_Dict_1.0
> ```
> python pu_main.py --type mPU --dataset CoNLL2003_Dict_1.0 --flag ALL --m 15 --lr 0.0005 --loss MAE --epochs 100
> ```


### For MPN:

> #### Example 1: train on BC5CDR_Dict_0.2
> ```
> python pu_main.py --type mPN --dataset BC5CDR_Dict_0.2 --flag ALL --m 28 --lr 0.0005 --loss MAE --embedding bio-embedding --epochs 100
> ```

> #### Example 2: train on CoNLL2003_Dict_0.2
> ```
> python pu_main.py --type mPN --dataset CoNLL2003_Dict_0.2 --flag ALL --m 15 --lr 0.0005 --loss MAE --epochs 100
> ```


### For BNPU:

> #### Example 1: train on BC5CDR_Dict_1.0
> ```
> Step 1: python pu_main.py --type bnPU --dataset BC5CDR_Dict_1.0 --flag Chemical --m 28 --embedding bio-embedding --epochs 100
> ```
> Step 1 is to train a binary classifier to determine entities of `Chemical` type.
> 
> ```
> Step 2: python pu_main.py --type bnPU --dataset BC5CDR_Dict_1.0 --flag Disease --m 28 --embedding bio-embedding --epochs 100
> ```
> Step 2 is to train a binary classifier to determine entities of `Disease` type.
> 
> ```
> Step 3: python pu_main.py --type bnPU --dataset BC5CDR_Dict_1.0 --inference True --embedding bio-embedding
> ```
> Step 3 is to infer the final prediction. You need to manually pass the path of saved models in the first 2 steps in `pu_main.py` (in the block of
> `if args.inference:`). 


> #### Example 2: train on CoNLL2003_Dict_1.0
> ```
> Step 1: python pu_main.py --type bnPU --dataset CoNLL2003_Dict_1.0 --flag PER --m 32 --epochs 100
> Step 2: python pu_main.py --type bnPU --dataset CoNLL2003_Dict_1.0 --flag LOC --m 25 --epochs 100
> Step 3: python pu_main.py --type bnPU --dataset CoNLL2003_Dict_1.0 --flag ORG --m 28 --epochs 100
> Step 4: python pu_main.py --type bnPU --dataset CoNLL2003_Dict_1.0 --flag MISC --m 62 --epochs 100
> Step 5: python pu_main.py --type bnPU --dataset CoNLL2003_Dict_1.0 --inference True
> ```
> Note: If use the same `m` value for all entity types of CoNLL2003 datasets, BNPU performs worse than individually setting the `m` value for each
 type.


## Citation
