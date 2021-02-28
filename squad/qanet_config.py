# QANet Config

# Data
glove_dim = 300 #Embedding dimension for Glove
char_dim = 64 #Embedding dimension for char
para_limit = 400 #Limit length for paragraph
ques_limit = 50 #Limit length for question

# Train
dropout = 0.1 #Dropout prob across the layers
dropout_char = 0.05 #Dropout prob across the layers
grad_clip = 5.0 #Global Norm gradient clipping rate
learning_rate = 0.001 #Learning rate
lr_warm_up_num = 1000 #Number of warm-up steps of learning rate
ema_decay = 0.9999 #Exponential moving average decay
beta1 = 0.8 #Beta 1
beta2 = 0.999 #Beta 2
early_stop = 10 #Checkpoints for early stop

# Model
d_model = 96 #Dimension of connectors of each layer
num_heads = 4 #Number of heads in multi-head attention
pretrained_char = True