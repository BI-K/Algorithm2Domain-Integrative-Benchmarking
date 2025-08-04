#######################################################################################
### Liquid Time Constant Network and closed-form Continuous-time Control (CfC) model ##
#######################################################################################

from ncps.torch import CfC, LTC
import torch.nn as nn


class LTCN(nn.Module):
    def __init__(self, configs):
        super(LTCN, self).__init__()
        self.input_size = configs.input_channels

        # return_sequences
        # ode_unfolds
        self.ode_unfolds = configs.ode_unfolds

        self.layers = configs.lctn_layers # list of number of units in each layer
        self.output_size = configs.features_len * configs.final_out_channels
        self.dropout = configs.dropout

        self.batch_norm = nn.BatchNorm1d(self.input_size)

        # for now no batchnorm layer

        self.ltc_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        self.ltc1 = LTC(self.input_size, self.layers[0])
        self.dropout1 = nn.Dropout(self.dropout)
        for layer in range(1, len(self.layers)):
            ltc_layer = LTC(self.layers[layer - 1], self.layers[layer])
            self.ltc_layers.append(ltc_layer)
            self.dropout_layers.append(nn.Dropout(self.dropout))

        self.ltc_last = LTC(self.layers[-1], self.output_size)
        self.final_layer = nn.Linear(self.output_size, self.output_size)

    def forward(self, x):

        # defualt batch_first = True
        # param input: Input tensor of shape (L,C) in batchless mode, or (B,L,C) if batch_first was set to True and (L,B,C) if batch_first is False
        # (batch_size, input_size, sequence_length) => (batch_size, sequence_length, input_size)
        x = x.transpose(2, 1)

        # optionally apply batch normalization
        if x.dim() == 3:
            x = x.transpose(1, 2)  # (batch_size, input_size, seq_len)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)  # (batch_size, seq_len, input_size)
        else:
            x = self.batch_norm(x)      

        ltc1_out, _ = self.ltc1(x)
        ltc1_out = self.dropout1(ltc1_out)
        for i in range(len(self.ltc_layers)):
            ltc_out, _ = self.ltc_layers[i](ltc1_out)
            ltc_out = self.dropout_layers[i](ltc_out)
            ltc1_out = ltc_out
        ltc1_out, _ = self.ltc_last(ltc1_out)

        last_output = self.final_layer(ltc1_out[:, -1, :])

        return last_output



class CfCN(nn.Module):
    def __init__(self, configs):
        super(CfCN, self).__init__()
        self.input_size = configs.input_channels

        # return_sequences
        # activation
        # back_bone_units
        self.backbone_units = configs.backbone_units
        self.backbone_layers = configs.backbone_layers
        self.backbone_dropout = configs.backbone_dropout
        

        self.layers = configs.cfcn_layers # list of number ofunits in each layer
        self.output_size = configs.features_len * configs.final_out_channels
        self.dropout = configs.dropout

        self.batch_norm = nn.BatchNorm1d(self.input_size)

        # for now no batchnorm layer

        self.ltc_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        self.ltc1 = CfC(self.input_size, self.layers[0], backbone_units=self.backbone_units, backbone_layers=self.backbone_layers, backbone_dropout=self.backbone_dropout)
        self.dropout1 = nn.Dropout(self.dropout)
        for layer in range(1, len(self.layers)):
            ltc_layer = CfC(self.layers[layer - 1], self.layers[layer], backbone_units=self.backbone_units, backbone_layers=self.backbone_layers, backbone_dropout=self.backbone_dropout)
            self.ltc_layers.append(ltc_layer)
            self.dropout_layers.append(nn.Dropout(self.dropout))

        self.ltc_last = CfC(self.layers[-1], self.output_size, backbone_units=self.backbone_units, backbone_layers=self.backbone_layers, backbone_dropout=self.backbone_dropout)
        self.final_layer = nn.Linear(self.output_size, self.output_size)

    def forward(self, x):

        # defualt batch_first = True
        # param input: Input tensor of shape (L,C) in batchless mode, or (B,L,C) if batch_first was set to True and (L,B,C) if batch_first is False
        # (batch_size, input_size, sequence_length) => (batch_size, sequence_length, input_size)
        x = x.transpose(2, 1)

        # optionally apply batch normalization
        if x.dim() == 3:
            x = x.transpose(1, 2)  # (batch_size, input_size, seq_len)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)  # (batch_size, seq_len, input_size)
        else:
            x = self.batch_norm(x)      

        ltc1_out, _ = self.ltc1(x)
        ltc1_out = self.dropout1(ltc1_out)
        for i in range(len(self.ltc_layers)):
            ltc_out, _ = self.ltc_layers[i](ltc1_out)
            ltc_out = self.dropout_layers[i](ltc_out)
            ltc1_out = ltc_out
        ltc1_out, _ = self.ltc_last(ltc1_out)
        last_output = self.final_layer(ltc1_out[:, -1, :])

        return last_output