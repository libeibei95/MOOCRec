# MOOCRec
*MOOCRec*: Disentangled Self-Supervision for Recommending Videos in MOOCs

Student: [*Abinash Sinha*](https://www.linkedin.com/in/abinashsinha330/)
<br>Mentor: Dr. Shalini Pandey
<br>Advisor: [*Dr. Jaideep Srivastava*](https://cse.umn.edu/cs/jaideep-srivastava)

Disentangled self-supervision is implemented as formulated in the paper, ["***Disentangled Self-Supervision in Sequential Recommenders"***](http://pengcui.thumedialab.com/papers/DisentangledSequentialRecommendation.pdf)  
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
explores the pre-processed data

## Data Preprocessing
```shell script
python ./data/data_preprocessor.py
```
data preprocessor to 
1. label encode video ids
2. calculate number of videos watched by each student
3. remove consecutive repetitions of video ids for each student (if asked to do)
3. perform negative sampling for testing purpose


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
python ./data/data_explorer.py
```
Total number of students: 48640


Number of courses watched: 3370213  
Average number of courses watched: 69.29  
Median number of courses watched: 35.0  
1st quartile number of courses watched: 16.0  
3rd quartile of courses watched: 81.0  
Maximum number of courses watched: 3377  
Minimum number of courses watched: 1  
Number of unique courses watched: 685  


Number of videos watched: 4663919  
Average number of videos watched: 95.89  
Median number of videos watched: 56.0  
1st quartile number of videos watched: 26.0  
3rd quartile of videos watched: 117.0  
Maximum number of videos watched: 3337  
Minimum number of videos watched: 2  
Number of unique videos watched: 34101  


## Pre-training (Disentangled Self-supervision)
```shell script
python pretrainer.py --data_name MOOCCube
```

## Training (Fine tuning)
We support two evaluation methods.

+ Rank ground-truth item with 99 randomly sampled negative items
```shell script
python finetuner.py --mode sample --data_name MOOCCube --ckp pretrain_epochs_num
```

+ Rank ground-truth item with all items
```shell script
python finetuner.py --mode full --data_name MOOCCube --ckp pretrain_epochs_num
```

## Explanation
```shell script
python explainer.py

```

### References
1. Jianxin Ma, Chang Zhou, Hongxia Yang, Peng Cui, Xin Wang, and Wenwu Zhu. 2020. 
**Disentangled Self-Supervision in Sequential Recommenders.** 
In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20). 
Association for Computing Machinery, New York, NY, USA, 483–491.
2. Wang-Cheng Kang and Julian McAuley. 2018. **Self-attentive sequential recommendation.** In ICDM 2018.
3. L. Kong, C. de Massond’Autume, L. Yu, W. Ling, Z. Dai, and D.Yogatama.2020. 
**A Mutual Information Maximization Perspective of Language Representation Learning.** In ICLR 2020.
4. A. van den Oord, Y. Li, and O. Vinyals. 2018. 
**Representation Learning with Contrastive Predictive Coding.** CoRR abs/1807.03748 (2018). arXiv:1807.03748
5. Kun Zhou, Hui Wang, Wayne Xin Zhao, Yutao Zhu, Sirui Wang, Fuzheng Zhang, Zhongyuan Wang, and Ji-Rong Wen. 2020. 
**S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization.** 
Proceedings of the 29th ACM International Conference on Information & Knowledge Management. 
Association for Computing Machinery, New York, NY, USA, 1893–1902. DOI:https://doi.org/10.1145/3340531.3411954

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
