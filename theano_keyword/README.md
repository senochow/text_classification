
Input format:
=====
json file:
Field Required:
* id: message id
* content
* words: [word/tag, ...]

Running 
=====
1. preprocessing
```
python process_data_message.py word2vec_file cv conf data_set

```

2. training
```
python conv_net_sentence.py cv data_set conf_file flag
```

3. prediction and generate keywords
```
python cnn_predict.py -nonstatic -word2vec cv flag data_set conf_file
```

4. generate top kegwords
```
python extract_keywords.py keywords_dir 1 >pos_keywords
```
