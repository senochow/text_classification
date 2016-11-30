Code for the paper [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) (EMNLP 2014).

Folk from yoonkom with slightly modified. 

Input data:
=====
* 2 files: one for pos and the other for neg.  each line is a text with words sperated by space

Running 
=====
1. First preprocess the data
python process_data.py word2vec_file pos_file neg_file

2. Run training
python conv_net_sentence.py -nonstatic -word2vec


