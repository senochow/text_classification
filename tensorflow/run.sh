
#w2v_file="/media/Data/zhouxing/data/word_vectors/GoogleNews-vectors-negative300.bin"
w2v_file="/media/Data/zhouxing/data/word_vectors/c_w2v_5kw_filter_weibo__iter_my_skip_sample_negative_10.vector"
pos_file="../data/zhihu/Yzhihu.json"
neg_file="../data/zhihu/Nzhihu.json"

python train.py $w2v_file $pos_file $neg_file


