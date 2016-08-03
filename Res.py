dataset_path = '.\\dataset\\imdb.pkl'

train_dataset_path = '.\\dataset\\train.pkl'
valid_dataset_path = '.\\dataset\\valid.pkl'
test_dataset_path = '.\\dataset\\test.pkl'

lstm_path = '.\\model\\lstm.pkl'
blstm_path = '.\\model\\blstm.pkl'
meanpooling_path = '.\\model\\meanpooling.pkl'

data_dim = 1
hiden_dim = 32
word_dim = 128
vocabulary_size = 100000
num_class = 2
max_seq_length = 100

batch_size = 90
test_batch_size = 10
valid_batch_size = 10

num_epoch = 100
lstm_grad_clip = 5
lstm_drop = 0.4
learning_rate = 0.5
floatX = 'float32'