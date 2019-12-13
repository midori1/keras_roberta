import os

os.environ['TF_KERAS'] = '1'
import tensorflow as tf
from keras_roberta.roberta import build_bert_model
from keras_roberta.tokenizer import RobertaTokenizer
from fairseq.models.roberta import RobertaModel as FairseqRobertaModel
import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--roberta_path", type=str, default='',
                        help="path to original roberta model")
    parser.add_argument("--tf_roberta_path", type=str, default='',
                        help="path to converted tf roberta model")
    parser.add_argument("--tf_ckpt_name", type=str, default='',
                        help="ckpt name")

    args = parser.parse_args()
    vocab_path = 'keras_roberta/'

    config_path = args.tf_roberta_path + '/bert_config.json'
    checkpoint_path = os.path.join(args.tf_roberta_path, args.tf_ckpt_name)

    gpt_bpe_vocab = vocab_path + 'encoder.json'
    gpt_bpe_merge = vocab_path + 'vocab.bpe'
    roberta_dict = args.roberta_path + '/roberta.base/dict.txt'

    tokenizer = RobertaTokenizer(gpt_bpe_vocab, gpt_bpe_merge, roberta_dict)
    model = build_bert_model(config_path, checkpoint_path, roberta=True,
                             return_all_hiddens=True)  # 建立模型，加载权重

    # 编码测试
    text = "你好我是中文"
    sep = [tokenizer.sep_token]
    cls = [tokenizer.cls_token]
    # 1. 先用'bpe_tokenize'将文本转换成bpe tokens
    tokens = tokenizer.bpe_tokenize(text)
    # 2. 然后自行添加一些标志token
    tokens = cls + tokens + sep + sep + tokens + sep
    # 3. 最后转换成id
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(token_ids)

    print('\n ===== tf model predicting =====\n')
    our_output = model.predict([np.array([token_ids]), np.array([segment_ids])])
    print(our_output)

    print('\n ===== torch model predicting =====\n')
    roberta = FairseqRobertaModel.from_pretrained(args.roberta_path)
    roberta.eval()  # disable dropout

    input_ids = roberta.encode(text, text).unsqueeze(0)  # batch of size 1
    their_output = roberta.model(input_ids, features_only=True)[0]
    print(their_output)

    # print('\n ===== reloading and predicting =====\n')
    # model.save('test.model')
    # del model
    # model = keras.models.load_model('test.model')
    # print(model.predict([np.array([token_ids]), np.array([segment_ids])]))

