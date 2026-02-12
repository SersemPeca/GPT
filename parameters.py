import torch

source_file_name = 'en_bg_data/train.bg'
target_file_name = 'en_bg_data/train.en'
source_dev_file_name = 'en_bg_data/dev.bg'
target_dev_file_name = 'en_bg_data/dev.en'

corpus_file_name = 'corpusData'
words_file_name = 'wordsData'
model_file_name = 'NMTmodel'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

d_model = 256
num_layers = 4
num_heads = 4

learning_rate = 0.0004
batch_size = 4
clip_grad = 1.0

max_epochs = 30
log_every = 10
test_every = 1000
