# Keras Roberta

Implement roberta based on [bert4keras](https://github.com/bojone/bert4keras)

## Requirments
```
fair-seq
bert4keras
```

## Usage

1. Convert roberta weights from PyTorch to Tensorflow

```
python convert_roberta_to_tf.py --model_name your_model_name
 --cache_dir /path/to/pytorch/roberta --tf_cache_dir /path/to/conveted/roberta
```

2. Extract features as `tf_roberta_demo.py`

```
python tf_roberta_demo.py --roberta_path /path/to/pytorch/roberta
--tf_roberta_path /path/to/converted/roberta
```
