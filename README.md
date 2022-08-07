# Revisiting Embedding-based Entity Alignment: A Robust and Adaptive Method

> Entity alignment—the discovery of identical entities across different knowledge graphs (KGs)—is a critical task in data fusion.
In this paper, we revisit existing entity alignment methods in practical and challenging scenarios. Our empirical studies show that current
work has a low level of robustness to long-tail entities and the lack of entity names or relation triples. We aim to develop a robust and
adaptive entity alignment method, and the availability of relations, attributes, or names is not required. Our method consists of an attribute
encoder and a relation encoder, representing an entity by aggregating its attributes or relational neighbors using the attention mechanisms
that can highlight the useful attributes and relations in end-to-end learning. To let the encoders complement each other and produce a
coherent representation space, we propose adaptive embedding fusion via a gate mechanism. We consider four entity alignment settings
for evaluation, i.e., the conventional setting with both relation and attribute triples, as well as three challenging settings without attributes,
without relations, without both relations and names, respectively. Results show that our method can achieve state-of-the-art performance.
Even in the most challenging setting without relations and names, our method can still achieve promising results while others fall.

## Dependencies

- TensorFlow==1.14
- [OpenEA](https://github.com/nju-websoft/OpenEA) 
- [Bert-as-service](https://bert-as-service.readthedocs.io/en/latest/section/get-start.html) (choose **BERT-Base, Uncased**)
- [fastText](https://fasttext.cc/docs/en/english-vectors.html) (optional) 

## Datasets

Create folder `datasets/` and put all datasets into it. A proper path should be like `datasets/DBP_en_DBP_fr_15K_V1/`. Datasets can be downloaded from [OpenEA](https://github.com/nju-websoft/OpenEA) .

## Running

**Before running this project, please start the bert-service first:**
```
bert-serving-start -model_dir ./multi_cased_L-12_H-768_A-12/ -num_worker=2
```

Enter the folder `src/`, and execute `run.sh` as follows. We've provided the default parameters for running the *RoadEA* model on the *DBP_en_DBP_fr_15K_V1* dataset. You can modify the hyper-parameters to run different modes and datasets.

```
$ bash run.sh
```

You can also manually enter a command line to run:

```
$ python main.py --model RoadEA --input ../datasets/DBP_en_DBP_fr_15K_V1/ 
```

> If you have any difficulty or question in running code and reproducing experimental results, please email to zqsun.nju@gmail.com or yuxinwangcs@outlook.com.

## Citation

```
@article{RoadEA,
  author    = {Zequn Sun and
               Wei Hu and
               Chengming Wang and
               Yuxin Wang and 
               Yuzhong Qu},
  title     = {Revisiting Embedding-based Entity Alignment: A Robust and Adaptive Method},
  journal   = {IEEE Transactions on Knowledge and Data Engineering},
  year      = {2022}
}
```
