# MOOCRec
*MOOCRec*: Disentangled Self-Supervision for Recommending Videos in MOOCs

Student: Abinash Sinha
<br>Mentor: Dr. Shalini Pandey
<br>Advisor: Dr. Jaideep Srivastava

Disentangled self-supervision is used as formulated in the paper, ["**Disentangled Self-Supervision in Sequential Recommenders"**](http://pengcui.thumedialab.com/papers/DisentangledSequentialRecommendation.pdf)  
Existing techniques used in sequential recommenders, usually adopt sequence-to-item training strategy, which supervises  a sequence model with a user’s next behavior as the label
and the user’s past behaviors as the input.  
Sequence-to-item training strategy is myopic and usually produces non-diverse recommendations. This paper addresses these challenges by adding a sequence-to-sequence training strategy
based on latent self-supervision and disentanglement. These two challenges basically imply:  
1. reconstructing a future sequence containing many behaviors is exponentially harder than reconstructing a single next behavior, which can lead to difficulty in convergence
2. sequence of all future behaviors can involve many intentions, not all of which may be predictable from the sequence of earlier behaviors.

## Requirements
```shell script
pip install -r requirements.txt
```

## Data Preprocessing
```shell script
data preprocessor to 
1. label encode video ids
2. calculate number of videos watched by each student
3. remove consecutive repetitions of video ids for each student (if asked to do)
3. perform negative sampling for testing purpose
./data/data_preprocessor.py
```

**MOOCCube.csv** has 3 columns:
1. *id*: student id
2. *video_ids*: comma-separated video ids
3. *num_video_ids*: number of videos watched by student  
For example,  

id | video_ids | num_video_ids
--- | --- | ---
user_1 | item_1,item_2,item_3 | 3
user_2 | item_1,item_2,item_3,item_4 | 4

**MOOCCube_sample.csv** has 2 columns:
1. *id*: student id
2. *video_ids*: comma-separated video ids (if 99 negative samples)
requested for each student  
For example,  

id | video_ids
--- | ---
user_1 | neg_item_1,neg_item_2,...,neg_item_99
user_2 | neg_item_1,neg_item_2,...,neg_item_99

## Data Exploration
```shell script
pip install -r requirements.txt
```

## Pre-training (Disentangled Self-supervision)
```shell script
python run_pretrain.py --data_name MOOCCube
```

## Training (Fine tuning)
We support two evaluation methods.

+ Rank ground-truth item with 99 randomly sampled negative items
```shell script
python run_train.py --neg_sampling sample --data_name MOOCCube --ckp pretrain_epochs_num
```

+ Rank ground-truth item with all items
```shell script
python run_train.py --neg_sampling full --data_name MOOCCube --ckp pretrain_epochs_num
```
### 

### Cite
If you find codes useful for your research or development, please cite:

```
              {2021-MOOCRec,
  author    = {Shalini Pandey and
               Abinash Sinha and
               Jaideep Srivastava},
  title     = {MOOCRec: Disentangled Self-Supervision for Recommending Videos in MOOCs},
  year      = {2021}
}
```

### Contact
If you have any questions for the codes, please send an email to sinha160@umn.edu.
