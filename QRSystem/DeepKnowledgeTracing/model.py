import torch.nn as nn


class DKTModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_skills, n_layers, dropout=0.6):
        super(DKTModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, dropout=dropout)
        self.n_hid = hidden_size
        self.n_layers = n_layers

        # nn.Linear是一个全连接层，hidden_size是输入层维数，num_skills是输出层维数
        # 因此，decoder是隐层(self.rnn)到输出层的网络
        self.decoder = nn.Linear(hidden_size, num_skills)

        self.init_weights()

    # 前向计算, 网络结构是：input --> hidden(self.rnn) --> decoder(输出层)
    def forward(self, input, hidden):
        # output: 隐藏层在各个时间步上计算并输出的隐藏状态, 形状是[时间步数, 批量大小, 隐层维数]
        output, hidden = self.lstm(input, hidden)
        # decoded: 形状是[时间步数, 批量大小, num_skills]
        decoded = self.decoder(output.contiguous().view(output.size(0) * output.size(1), output.size(2)))
        return decoded, hidden

    def init_weights(self):
        initrange = 0.05
        # 隐层到输出层的网络的权重
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters())

        return (weight.new_zeros(self.n_layers, bsz, self.n_hid),
                weight.new_zeros(self.n_layers, bsz, self.n_hid))


