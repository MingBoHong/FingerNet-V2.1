# Hyperparameter
growth_k = 12
nb_block = 2 # how many (dense block + Transition Layer) ?
dropout_rate = 0.5
class_num = 2

# Momentum Optimizer will use
nesterov_momentum = 0.9
weight_decay = 1e-4

# Label & batch_size
batch_size = 128


checkpoint_path = 'model/checkpoint'  # 设置模型参数文件所在路径
event_log_path = 'event-log'  # 设置事件文件所在路径，用于周期性存储Summary缓存对象
Trainrecords = 'train.tfrecords'
Testrecords = 'Test.tfrecords'
epochs = 3

l2 = 0.0001
decay_step = 20000               # 衰减迭代数
learning_rate_decay_factor = 0.1  # 学习率衰减因子
initial_learning_rate = 0.1      # 初始学习率

pic_size=32

init_learning_rate = 1e-3
epsilon = 1e-8 # AdamOptimizer epsilon