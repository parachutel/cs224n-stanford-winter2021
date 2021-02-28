"""QANet Config

Refer to:
    https://github.com/andy840314/QANet-pytorch-/blob/master/config.py
    https://github.com/heliumsea/QANet-pytorch/blob/master/config.py
"""

# Data
glove_dim = 300 #Embedding dimension for Glove
char_dim = 64 #Embedding dimension for char
para_limit = 400 #Limit length for paragraph
ques_limit = 50 #Limit length for question

# Train
dropout = 0.1 #Dropout prob across the layers
dropout_char = 0.05 #Dropout prob across the layers
learning_rate = 0.001 #Learning rate
lr_warm_up_num = 1000 #Number of warm-up steps of learning rate

# Model
d_model = 96 #Dimension of connectors of each layer
num_heads = 4 #Number of heads in multi-head attention
pretrained_char = True