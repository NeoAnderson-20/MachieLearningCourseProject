# 导入需要的库
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tushare as ts
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import TensorDataset
from tqdm import tqdm
import math
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)

# 0. 参数配置
class Config():
    data_path = './ETTh1.csv'
    data_path_new = './ETTh1_new_smaller_dataset.csv'
    data_path_train = './train_set.csv'
    data_path_validation = './validation_set.csv'
    data_path_test = './test_set.csv'
    timestep = 336  # 时间步长，就是利用多少时间窗口
    batch_size = 96  # 批次大小
    feature_size = 7  # 每个步长对应的特征数量，这里只使用1维，每天的风速
    hidden_size = 256  # 隐层大小
    output_size = 336  # 由于是单输出任务，最终输出层大小为1，预测未来1天风速,相当于输出时间步
    num_layers = 2  # lstm的层数
    transformer_num_layers = 1 # transformer层数
    epochs = 100 # 迭代轮数
    best_loss = 999 # 记录损失
    learning_rate = 3e-5 # 学习率，之前是3e-5
    model_name = 'transformer' # 模型名称
    save_path = './{}.pth'.format(model_name) # 最优模型保存路径
config = Config()

# 1. 加载时间序列数据
df_total = pd.read_csv(config.data_path, index_col=0)
df_train = pd.read_csv(config.data_path_train, index_col=0)
print(df_train)
df_validation = pd.read_csv(config.data_path_validation, index_col=0)
df_test = pd.read_csv(config.data_path_test, index_col=0)

# 2. 对数据进行标准化
# scaler = MinMaxScaler()
scaler_model = MinMaxScaler()
scaler_train = MinMaxScaler()
scaler_validation = MinMaxScaler()
scaler_test = MinMaxScaler()
data_total = scaler_model.fit_transform(np.array(df_total))
data_train = scaler_train.fit_transform(np.array(df_train))
data_validation = scaler_validation.fit_transform(np.array(df_validation))
data_test = scaler_test.fit_transform(np.array(df_test))


# 3. 形成训练数据
def split_data(data, timestep, feature_size, output_size):
    dataX = []  # 保存X
    dataY = []  # 保存Y

    # 将整个窗口的数据保存到X中，将未来预测保存到Y中
    for index in range(len(data) - timestep - 1):
        # find the end of this pattern
        end_ix = index + timestep
        out_end_ix = end_ix + output_size
        # check if we are beyond the dataset
        if out_end_ix > len(data):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = data[index: end_ix, :], data[end_ix:out_end_ix, :]
        dataX.append(seq_x)
        dataY.append(seq_y)

        '''
        dataX.append(data[index: index + timestep, :])
        # dataY是第0列变量数据，在给定的时间步窗口
        # dataY.append(data[index + timestep : index + timestep + output_size][:, 0].tolist())
        # 如果考虑多步，那么dataY也应该是一个三维数组，可以按如下设置
        #dataY.append(data[index + timestep : index + timestep + output_size, :])
        y_values = data[index + timestep: index + timestep + output_size][:, 0].tolist()

        # 如果长度不足 output_size，可以选择填充缺失的值
        while len(y_values) < output_size:
            y_values.append(0)  # 或者使用其他填充值

        dataY.append(y_values)

    output_file = 'output4.csv'

    # 将多个子列表写入CSV文件
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(dataY)
    '''
    # dataX是三维，dataY是三维
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    print("dataX的形状是：", dataX.shape)
    print("dataY的形状是：", dataY.shape)

    x_train = dataX.reshape(-1, timestep, feature_size)
    y_train = dataY.reshape(-1, output_size, feature_size)

    return [x_train, y_train]


# 4. 获取训练数据
x_train, y_train = split_data(data_train, config.timestep, config.feature_size, config.output_size)
x_valida, y_valida = split_data(data_validation, config.timestep, config.feature_size, config.output_size)
x_test, y_test = split_data(data_test, config.timestep, config.feature_size, config.output_size)
print("x_train的形状是：", x_train.shape)
print("y_train的形状是：", y_train.shape)
print("x_valida的形状是：", x_valida.shape)
print("y_valida的形状是：", y_valida.shape)
print("x_test的形状是：", x_test.shape)
print("y_test的形状是：", x_test.shape)

# 5.数据转化为tensor
x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
reverse_x_train_tensor = scaler_model.inverse_transform(x_train[1])
y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
reverse_y_train_tensor = scaler_model.inverse_transform(y_train[0])

x_valida_tensor = torch.from_numpy(x_valida).to(torch.float32)
y_valida_tensor = torch.from_numpy(y_valida).to(torch.float32)
x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
y_test_tensor = torch.from_numpy(y_test).to(torch.float32)
reverse_y_test_tensor = scaler_model.inverse_transform(y_test[0])

# 6.形成训练数据集
train_data = TensorDataset(x_train_tensor, y_train_tensor)
valida_data = TensorDataset(x_valida_tensor, y_valida_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor)

# 7.将数据加载成迭代器
train_loader = torch.utils.data.DataLoader(train_data,
                                           config.batch_size,
                                           False, drop_last=True)
valida_loader = torch.utils.data.DataLoader(valida_data,
                                            config.batch_size,
                                            False)

test_loader = torch.utils.data.DataLoader(test_data,
                                          config.batch_size,
                                          False)

# 8. 定义位置编码模型
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        chunk = x.chunk(x.size(-1), dim=2)
        out = torch.Tensor([]).to(x.device)
        for i in range(len(chunk)):
            out = torch.cat((out, chunk[i] + self.pe[:chunk[i].size(0), ...]), dim=2)
        return out

# 9.生成 Transformer 模型中的目标序列掩码
def transformer_generate_tgt_mask(length, device):
    mask = torch.tril(torch.ones(length, length, device=device)) == 1
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask

# 10. 定义transformer模型类
class Transformer(nn.Module):
    """标准的Transformer编码器-解码器结构"""

    def __init__(self, n_encoder_inputs, n_decoder_inputs, Sequence_length, d_model=512, dropout=0.1, num_layer=8):
        """
        初始化
        :param n_encoder_inputs:    输入数据的特征维度
        :param n_decoder_inputs:    编码器输入的特征维度，其实等于编码器输出的特征维度
        :param d_model:             词嵌入特征维度
        :param dropout:             dropout
        :param num_layer:           Transformer块的个数
         Sequence_length:           transformer 输入数据 序列的长度
        """
        super(Transformer, self).__init__()

        self.input_pos_embedding = torch.nn.Embedding(5000, embedding_dim=d_model)
        self.target_pos_embedding = torch.nn.Embedding(5000, embedding_dim=d_model)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=dropout,
                                                         dim_feedforward=4 * d_model)
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=d_model, nhead=8, dropout=dropout,
                                                         dim_feedforward=4 * d_model)

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.input_projection = torch.nn.Linear(n_encoder_inputs, d_model)
        self.output_projection = torch.nn.Linear(n_decoder_inputs, d_model)

        self.linear = torch.nn.Linear(d_model, 1)
        self.ziji_add_linear = torch.nn.Linear(Sequence_length, 336* 7)


    def encode_in(self, src):
        src_start = self.input_projection(src).permute(1, 0, 2)
        in_sequence_len, batch_size = src_start.size(0), src_start.size(1)
        pos_encoder = (torch.arange(0, in_sequence_len, device=src.device).unsqueeze(0).repeat(batch_size, 1))
        pos_encoder = self.input_pos_embedding(pos_encoder).permute(1, 0, 2)
        src = src_start + pos_encoder
        src = self.encoder(src) + src_start
        return src

    def decode_out(self, tgt, memory):
        tgt_start = self.output_projection(tgt).permute(1, 0, 2)
        out_sequence_len, batch_size = tgt_start.size(0), tgt_start.size(1)
        pos_decoder = (torch.arange(0, out_sequence_len, device=tgt.device).unsqueeze(0).repeat(batch_size, 1))
        pos_decoder = self.target_pos_embedding(pos_decoder).permute(1, 0, 2)
        tgt = tgt_start + pos_decoder
        tgt_mask = transformer_generate_tgt_mask(out_sequence_len, tgt.device)
        out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask) + tgt_start
        out = out.permute(1, 0, 2)  # [batch_size, seq_len, d_model]
        out = self.linear(out)
        return out

    def forward(self, src, target_in):
        # print("src.shape", src.shape)
        src = self.encode_in(src).to(device)
        # print("src.shape",src.shape)#src.shape torch.Size([9, 8, 512])
        out = self.decode_out(tgt=target_in, memory=src).to(device)
        # print("out.shape",out.shape)
        # print("out.shape:",out.shape)# torch.Size([batch, 3, 1]) # 原本代码中的输出
        # 上边的这个输入可以用于很多任务的输出 可以根据任务进行自由的变换
        # 下面是自己修改的
        # 使用全连接变成 [batch,1] 构成了基于transformer的回归单值预测
        out = out.squeeze(2)
        out = self.ziji_add_linear(out)
        out = out.view(config.batch_size, 336, 7)
        return out

# 11. 创建Transformer 模型，设置损失函数和优化器
model = Transformer(n_encoder_inputs=7, n_decoder_inputs=7, Sequence_length=336).to(device)  # 3 表示Sequence_length  transformer 输入数据 序列的长度

# model = Transformer(config.hidden_size, config.num_layers, config.feature_size, config.output_size,
#                     transformer_num_layers=config.transformer_num_layers)
loss_function = nn.L1Loss().to(device)  # 定义损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)  # 定义优化器


# 12.模型训练
y_list = []
for epoch in range(config.epochs):
    model.train()
    running_loss = 0
    train_bar = tqdm(train_loader)  # 形成进度条
    for data in train_bar:
        x_train, y_train = data  # 解包迭代器中的X和Y
        # print(x_train.shape)
        # print(y_train.shape)
        optimizer.zero_grad()
        tgt_in = torch.rand((config.batch_size, 336, 7)).to(device)
        y_train_pred = model(x_train.to(device), y_train.to(device))
        #y_list += torch.tensor(y_train_pred, requires_grad= False)
        # print(y_train_pred.shape)
        # print(y_train.shape)
        loss = loss_function(y_train_pred, y_train.to(device))
        #print("loss是：{:.3f}".format(loss))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if loss < config.best_loss:
            config.best_loss = loss
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 config.epochs,
                                                                 loss)
'''
    # 模型验证
    model.eval()
    valida_loss = 0
    with torch.no_grad():
        valida_bar = tqdm(valida_loader)
        for data in valida_bar:
            x_valida, y_valida = data
            y_valida_pred = model(x_valida)
            valida_loss = loss_function(y_valida_pred, y_valida)
            valida_bar.desc = "val loss:{:.3f}".format(valida_loss)
    if valida_loss < config.best_loss:
        config.best_loss = valida_loss
        torch.save(model, config.save_path)
'''

print('Finished Training')

# 13. 训练集验证
device = 'cpu'
model.to(device)

plot_size = 200
plt.figure(figsize=(12, 8))
# model = torch.load(config.save_path)
train_bar = tqdm(train_loader)  # 形成进度条
my_list1 = []
my_list2 = []
flag = 0
for data in train_bar:
    x_train, y_train = data  # 解包迭代器中的X和Y
    flag = flag + 1
    # y_train_pred = model(x_train, tgt_in)
    tgt_in = torch.rand((config.batch_size, 336, 7)).to(device)
    y_train_pred = model(x_train.to(device), y_train.to(device))
    # print("看一下y_train_pred的维度：", y_train_pred.shape)
    # print(y_train_pred)
    # print("看一下y_train_tensor的维度：", y_train_tensor.shape)
    # print(y_train_tensor[0])

    # plt.plot((y_train_pred.detach().numpy()[0])[:,-1].reshape(-1, 1), "b")
    # plt.plot((y_train_tensor.detach().numpy()[0])[:,-1].reshape(-1, 1), "r")
    # plt.legend()
    # plt.show()

    '''
    if flag == 1:
        my_list1.extend(scaler_model.inverse_transform((y_train_tensor.detach()).numpy()[0])[:,-1].reshape(-1, 1).flatten())

    if flag > 2:
        my_list2.extend(scaler_model.inverse_transform((y_train_pred.detach()).numpy()[127])[:,-1].reshape(-1, 1).flatten())
        #注意区分下：这里应该是y_train.detach()这个局部变量，而不是之前写的y_train_tensor.detach()这个全局变量
        my_list1.extend(scaler_model.inverse_transform((y_train.detach()).numpy()[127])[:,-1].reshape(-1, 1).flatten())
        print("my_list1的前336部分:\n",my_list1[:336])
        print("my_list1的后336部分:\n",my_list1[336:])
        print("这是长度为672的list1:\n", my_list1)
        print("这是长度为336的list2:\n", my_list2)
        plt.plot(range(336, 672), my_list2, "b", label = 'Prediction')
        plt.plot(my_list1, "orange", label = 'GroundTruth')
        plt.legend()
        plt.show()
        break

    '''

    my_list1.extend(scaler_train.inverse_transform((y_train_pred.detach()).numpy()[95])[:, -1].reshape(-1, 1).flatten())
    plt.plot(range(336, 672), my_list1, "b", label='Prediction')

    my_list2.extend(
        scaler_train.inverse_transform((y_train_tensor.detach()).numpy()[0])[:, -1].reshape(-1, 1).flatten())
    my_list2.extend(
        scaler_train.inverse_transform((y_train_tensor.detach()).numpy()[95])[:, -1].reshape(-1, 1).flatten())
    plt.plot(my_list2, "orange", label='GroundTruth')
    plt.legend()
    plt.show()
    break

    '''
    plt.plot(scaler_train.inverse_transform((y_train_pred.detach()).numpy()[96])[:, -1].reshape(-1, 1), "b")
    plt.plot(scaler_train.inverse_transform((y_train_tensor.detach()).numpy()[96])[:, -1].reshape(-1, 1), "r")

    # plt.plot(scaler.inverse_transform((y_train_pred.detach().numpy()[0][:, 0]).reshape(-1, 1)), "b")
    # plt.plot(scaler.inverse_transform((torch.Tensor(y_list).detach().numpy()[0][:, 0]).reshape(-1, 1)), "b")
    # plt.plot(scaler.inverse_transform((y_train_tensor.detach().numpy()[0][:, 0]).reshape(-1, 1)), "r")
    # plt.plot(scaler.inverse_transform(y_train_tensor.detach(|).numpy().reshape(-1, 1)[: plot_size]), "r")
    # plt.plot(scaler.inverse_transform((y_train_tensor.detach().numpy()[: plot_size, 0]).reshape(-1, 1)), "r")
    plt.legend()
    plt.show()
    break'''

# 14. 测试集测试验证
test_bar = tqdm(test_loader)  # 形成进度条
my_list3 = []
my_list4 = []
flag2 = 0
for data in test_bar:
    x_test, y_test = data  # 解包迭代器中的X和Y
    flag2 = flag2 + 1
    tgt_in = torch.rand((config.batch_size, 336, 7)).to(device)
    y_test_pred = model(x_test.to(device), y_test.to(device))
    plt.figure(figsize=(12, 8))
    print("看一下y_test_pred的维度：", y_test_pred.shape)
    # print(y_test_pred)
    print("看一下y_test_tensor的维度：", y_test_tensor.shape)
    # print(y_test_tensor)

    '''
    if flag2 == 1:
        my_list3.extend(scaler_model.inverse_transform((y_test_tensor.detach()).numpy()[0])[:,-1].reshape(-1, 1).flatten())

    if flag2 > 2:
        my_list4.extend(scaler_model.inverse_transform((y_test_pred.detach()).numpy()[127])[:,-1].reshape(-1, 1).flatten())
        #注意区分下：这里应该是y_test.detach()这个局部变量，而不是之前写的y_test_tensor.detach()这个全局变量
        my_list3.extend(scaler_model.inverse_transform((y_test.detach()).numpy()[127])[:,-1].reshape(-1, 1).flatten())
        plt.plot(range(336, 672), my_list4, "b", label = 'Prediction')
        plt.plot(my_list3, "orange", label = 'GroundTruth')
        plt.legend()
        plt.show()
        break
    '''

    my_list3.extend(scaler_test.inverse_transform((y_test_pred.detach()).numpy()[95])[:, -1].reshape(-1, 1).flatten())
    plt.plot(range(336, 672), my_list3, "b", label='Prediction')

    my_list4.extend(scaler_test.inverse_transform((y_test_tensor.detach()).numpy()[0])[:, -1].reshape(-1, 1).flatten())
    my_list4.extend(scaler_test.inverse_transform((y_test_tensor.detach()).numpy()[95])[:, -1].reshape(-1, 1).flatten())
    plt.plot(my_list4, "orange", label='GroundTruth')
    plt.legend()
    plt.show()
    break

    '''
    plt.plot(scaler_test.inverse_transform((y_test_pred.detach()).numpy()[96])[:, -1].reshape(-1, 1), "b")
    plt.plot(scaler_test.inverse_transform((y_test_tensor.detach()).numpy()[96])[:, -1].reshape(-1, 1), "r")

    # plt.plot(scaler.inverse_transform((y_test_pred.detach().numpy()[0][:, 0]).reshape(-1, 1)), "b")
    # plt.plot(scaler.inverse_transform((y_test_tensor.detach().numpy()[0][:, 0]).reshape(-1, 1)), "r")
    # plt.plot(scaler.inverse_transform(y_test_pred.detach().numpy()[: plot_size, 0].reshape(-1, 1)), "b")
    # plt.plot(scaler.inverse_transform(y_test_tensor.detach().numpy().reshape(-1, 1)[: plot_size]), "r")
    # plt.plot(scaler.inverse_transform((y_test_tensor.detach().numpy()[: plot_size, 0]).reshape(-1, 1)), "r")
    plt.legend()
    plt.show()
    break'''