
w2v_file="/media/Data/zhouxing/data/word_vectors/GoogleNews-vectors-negative300.bin"
pos_file="../data/rt-polaritydata/rt-polarity.pos"
neg_file="../data/rt-polaritydata/rt-polarity.neg"

python train.py $w2v_file $pos_file $neg_file


