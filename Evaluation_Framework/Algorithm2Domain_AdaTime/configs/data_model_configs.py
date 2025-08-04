def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class HAR():
    def __init__(self,
                scenarios,
                kernel_size=3,
                stride=1,
                dropout=0.5,
                normalize=True,
                shuffle=True,
                drop_last=True,
                mid_channels=64,
                final_out_channels=128,
                features_len=1,
                tcn_layers=[75, 150],
                tcn_kernel_size=17,
                tcn_dropout=0.5,
                lstm_hid=128,
                lstm_n_layers=1,
                lstm_bid=False,
                disc_hid_dim=64,
                DSKN_disc_hid=128,
                hidden_dim=500):
        super(HAR, self)
        # dataset parameters
        self.scenarios = [("2", "11")]
        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
        self.num_cont_output_channels = 0
        self.sequence_len = 128
        self.num_classes = 6
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # Model configs
        self.input_channels = 6
        self.kernel_size = 32
        self.stride = 6
        self.dropout = 0.1

        self.mid_channels = 64
        # d_model
        self.final_out_channels = 64
        self.features_len = 1

        # SWIFT features
        self.lstm_layers = [256, 16]
        self.bidirectional_lstm = True

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # LTCN features
        self.lctn_layers = [64, 32]
        self.ode_unfolds = 2

        # CfCN features 
        self.cfcn_layers = [64, 32]
        self.backbone_units = 32
        self.backbone_layers = 1
        self.backbone_dropout = 0.1


        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False


        # hinrichs GRU features
        self.hidden_size_gru = 64
        # shared between gru and transformer
        self.num_layers = 1

        # hinrichs Transformer
        self.pos_encoding = 'fixed'
        # TODO possibly in hparam - could be expanded to the other models
        self.activation='gelu'
        self.norm = 'BatchNorm'
        self.freeze = False
        self.n_heads = 8
        self.dim_feedforward = 128

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500

        self.data_augmentation_configs = None
        
        
class EEG():
    def __init__(self,
                scenarios,
                kernel_size=3,
                stride=1,
                dropout=0.5,
                normalize=True,
                shuffle=True,
                drop_last=True,
                mid_channels=64,
                final_out_channels=128,
                features_len=1,
                tcn_layers=[75, 150],
                tcn_kernel_size=17,
                tcn_dropout=0.5,
                lstm_hid=128,
                lstm_n_layers=1,
                lstm_bid=False,
                disc_hid_dim=64,
                DSKN_disc_hid=128,
                hidden_dim=500):
        super(EEG, self).__init__()
        # data parameters
        self.num_classes = 5
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        self.num_cont_output_channels = 0
        self.sequence_len = 3000
        self.scenarios = [("0", "11"), ("7", "18"), ("9", "14"), ("12", "5"), ("16", "1"),
                          ("3", "19"), ("18", "12"), ("13", "17"), ("5", "15"), ("6", "2")]
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # Model configs
        self.input_channels = 6
        self.kernel_size = 32
        self.stride = 6
        self.dropout = 0.1

        self.mid_channels = 64
        # d_model
        self.final_out_channels = 64
        self.features_len = 1

        # SWIFT features
        self.lstm_layers = [256, 16]
        self.bidirectional_lstm = True

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # LTCN features
        self.lctn_layers = [64, 32]
        self.ode_unfolds = 2

        # CfCN features 
        self.cfcn_layers = [64, 32]
        self.backbone_units = 32
        self.backbone_layers = 1
        self.backbone_dropout = 0.1


        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False


        # hinrichs GRU features
        self.hidden_size_gru = 64
        # shared between gru and transformer
        self.num_layers = 1

        # hinrichs Transformer
        self.pos_encoding = 'fixed'
        # TODO possibly in hparam - could be expanded to the other models
        self.activation='gelu'
        self.norm = 'BatchNorm'
        self.freeze = False
        self.n_heads = 8
        self.dim_feedforward = 128

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500

        self.data_augmentation_configs = None

class WISDM(object):
    def __init__(self,
                scenarios,
                kernel_size=3,
                stride=1,
                dropout=0.5,
                normalize=True,
                shuffle=True,
                drop_last=True,
                mid_channels=64,
                final_out_channels=128,
                features_len=1,
                tcn_layers=[75, 150],
                tcn_kernel_size=17,
                tcn_dropout=0.5,
                lstm_hid=128,
                lstm_n_layers=1,
                lstm_bid=False,
                disc_hid_dim=64,
                DSKN_disc_hid=128,
                hidden_dim=500):
        super(WISDM, self).__init__()
        self.class_names = ['walk', 'jog', 'sit', 'stand', 'upstairs', 'downstairs']
        self.sequence_len = 128
        self.scenarios = [("7", "18"), ("20", "30"), ("35", "31"), ("17", "23"), ("6", "19"),
                          ("2", "11"), ("33", "12"), ("5", "26"), ("28", "4"), ("23", "32")]
        self.num_classes = 6
        self.num_cont_output_channels = 0
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # Model configs
        self.input_channels = 6
        self.kernel_size = 32
        self.stride = 6
        self.dropout = 0.1

        self.mid_channels = 64
        # d_model
        self.final_out_channels = 64
        self.features_len = 1

        # SWIFT features
        self.lstm_layers = [256, 16]
        self.bidirectional_lstm = True

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # LTCN features
        self.lctn_layers = [64, 32]
        self.ode_unfolds = 2

        # CfCN features 
        self.cfcn_layers = [64, 32]
        self.backbone_units = 32
        self.backbone_layers = 1
        self.backbone_dropout = 0.1


        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False


        # hinrichs GRU features
        self.hidden_size_gru = 64
        # shared between gru and transformer
        self.num_layers = 1

        # hinrichs Transformer
        self.pos_encoding = 'fixed'
        # TODO possibly in hparam - could be expanded to the other models
        self.activation='gelu'
        self.norm = 'BatchNorm'
        self.freeze = False
        self.n_heads = 8
        self.dim_feedforward = 128

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500

        self.data_augmentation_configs = None

class OHIO(object):
    def __init__(self, 
                class_names=["Not Hypoglycemic", "Hypoglycemic"],
                scenarios=[("0", "1")],
                num_classes=2,
                input_channels=1,
                kernel_size=3,
                stride=1,
                dropout=0.5,
                normalize=True,
                shuffle=True,
                drop_last=True,
                mid_channels=64,
                final_out_channels=128,
                features_len=1,
                tcn_layers=[75, 150],
                tcn_kernel_size=17,
                tcn_dropout=0.5,
                lstm_hid=128,
                lstm_n_layers=1,
                lstm_bid=False,
                disc_hid_dim=64,
                DSKN_disc_hid=128,
                hidden_dim=500):
        super(OHIO, self).__init__()
        self.class_names = class_names
        self.scenarios = scenarios

        # Model configs
        self.input_channels = 6
        self.kernel_size = 32
        self.stride = 6
        self.dropout = 0.1

        self.mid_channels = 64
        # d_model
        self.final_out_channels = 64
        self.features_len = 1

        # SWIFT features
        self.lstm_layers = [256, 16]
        self.bidirectional_lstm = True

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # LTCN features
        self.lctn_layers = [64, 32]
        self.ode_unfolds = 2

        # CfCN features 
        self.cfcn_layers = [64, 32]
        self.backbone_units = 32
        self.backbone_layers = 1
        self.backbone_dropout = 0.1


        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False


        # hinrichs GRU features
        self.hidden_size_gru = 64
        # shared between gru and transformer
        self.num_layers = 1

        # hinrichs Transformer
        self.pos_encoding = 'fixed'
        # TODO possibly in hparam - could be expanded to the other models
        self.activation='gelu'
        self.norm = 'BatchNorm'
        self.freeze = False
        self.n_heads = 8
        self.dim_feedforward = 128

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500

        self.data_augmentation_configs = None

class WEATHER(object):
    def __init__(self,
                 class_names=["Below 25.5°C", "25.5-26.5°C", "26.5-27.5°C", "27.5-28.5°C",
                              "28.5-29.5°C", "29.5-30.5°C", "30.5-31.5°C", "Above 31.5°C"],
                 scenarios=[(1, 2), (0, 2), (4, 2), (2, 1)],
                 num_classes=8,
                 input_channels=7,
                 kernel_size=3,
                 stride=1,
                 dropout=0.5,
                 normalize=True,
                 shuffle=True,
                 drop_last=True,
                 mid_channels=64,
                 final_out_channels=128,
                 features_len=12,
                 tcn_layers=[75, 150],
                 tcn_kernel_size=17,
                 tcn_dropout=0.5,
                 lstm_hid=128,
                 lstm_n_layers=1,
                 lstm_bid=False,
                 disc_hid_dim=64,
                 DSKN_disc_hid=128,
                 hidden_dim=500
                 ):
        super(WEATHER, self).__init__()
        # Temperature classes for discretization
        self.class_names = class_names
        # (1, 2) - Madrid with Bilbao
        # (0, 2) - Valencia with Bilbao
        # (4, 2) - Seville with Bilbao
        # (2, 1) - Bilbao with Madrid
        self.scenarios = scenarios

        # Model configs
        self.input_channels = 6
        self.kernel_size = 32
        self.stride = 6
        self.dropout = 0.1

        self.mid_channels = 64
        # d_model
        self.final_out_channels = 64
        self.features_len = 1

        # SWIFT features
        self.lstm_layers = [256, 16]
        self.bidirectional_lstm = True

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # LTCN features
        self.lctn_layers = [64, 32]
        self.ode_unfolds = 2

        # CfCN features 
        self.cfcn_layers = [64, 32]
        self.backbone_units = 32
        self.backbone_layers = 1
        self.backbone_dropout = 0.1


        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False


        # hinrichs GRU features
        self.hidden_size_gru = 64
        # shared between gru and transformer
        self.num_layers = 1

        # hinrichs Transformer
        self.pos_encoding = 'fixed'
        # TODO possibly in hparam - could be expanded to the other models
        self.activation='gelu'
        self.norm = 'BatchNorm'
        self.freeze = False
        self.n_heads = 8
        self.dim_feedforward = 128

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500

        self.data_augmentation_configs = None

class HHAR(object):  ## HHAR dataset, SAMSUNG device.
    def __init__(self,
                 sequence_len=128,
                 scenarios=[("0", "6")],
                 class_names=['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down'],
                 num_classes=6,
                 shuffle=True,
                 drop_last=True,
                 normalize=True,
                 input_channels=3,
                 kernel_size=5,
                 stride=1,
                 dropout=0.5,
                 mid_channels=64,
                 final_out_channels=128,
                 features_len=1,
                 tcn_layers=[75, 150],
                 tcn_kernel_size=17,
                 tcn_dropout=0.0,
                 lstm_hid=128,
                 lstm_n_layers=1,
                 lstm_bid=False,
                 disc_hid_dim=64,
                 DSKN_disc_hid=128,
                 hidden_dim=500,
                 hidden_size = 32,
                 ode_solver_unfolds = 6
                 ):
        super(HHAR, self).__init__()
        self.sequence_len = sequence_len
        self.scenarios = scenarios
        self.class_names = class_names
        self.num_classes = num_classes

        self.num_cont_output_channels = 0

        self.shuffle = shuffle
        self.drop_last = drop_last
        self.normalize = normalize
        self.data_augmentation_configs = None

        # Model configs
        self.input_channels = 6
        self.kernel_size = 32
        self.stride = 6
        self.dropout = 0.1

        self.mid_channels = 64
        # d_model
        self.final_out_channels = 64
        self.features_len = 1

        # SWIFT features
        self.lstm_layers = [256, 16]
        self.bidirectional_lstm = True

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # LTCN features
        self.lctn_layers = [64, 32]
        self.ode_unfolds = 2

        # CfCN features 
        self.cfcn_layers = [64, 32]
        self.backbone_units = 32
        self.backbone_layers = 1
        self.backbone_dropout = 0.1


        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False


        # hinrichs GRU features
        self.hidden_size_gru = 64
        # shared between gru and transformer
        self.num_layers = 1

        # hinrichs Transformer
        self.pos_encoding = 'fixed'
        # TODO possibly in hparam - could be expanded to the other models
        self.activation='gelu'
        self.norm = 'BatchNorm'
        self.freeze = False
        self.n_heads = 8
        self.dim_feedforward = 128

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500

        self.data_augmentation_configs = None

        
        
class FD(object):
    def __init__(self,
                scenarios,
                kernel_size=3,
                stride=1,
                dropout=0.5,
                normalize=True,
                shuffle=True,
                drop_last=True,
                mid_channels=64,
                final_out_channels=128,
                features_len=1,
                tcn_layers=[75, 150],
                tcn_kernel_size=17,
                tcn_dropout=0.5,
                lstm_hid=128,
                lstm_n_layers=1,
                lstm_bid=False,
                disc_hid_dim=64,
                DSKN_disc_hid=128,
                hidden_dim=500):
        super(FD, self).__init__()
        self.sequence_len = 5120
        self.scenarios = [("0", "1"), ("0", "3"), ("1", "0"), ("1", "2"),("1", "3"),
                          ("2", "1"),("2", "3"),  ("3", "0"), ("3", "1"), ("3", "2")]
        self.class_names = ['Healthy', 'D1', 'D2']
        self.num_classes = 3

        self.num_cont_output_channels = 0

        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # Model configs
        self.input_channels = 6
        self.kernel_size = 32
        self.stride = 6
        self.dropout = 0.1

        self.mid_channels = 64
        # d_model
        self.final_out_channels = 64
        self.features_len = 1

        # SWIFT features
        self.lstm_layers = [256, 16]
        self.bidirectional_lstm = True

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # LTCN features
        self.lctn_layers = [64, 32]
        self.ode_unfolds = 2

        # CfCN features 
        self.cfcn_layers = [64, 32]
        self.backbone_units = 32
        self.backbone_layers = 1
        self.backbone_dropout = 0.1


        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False


        # hinrichs GRU features
        self.hidden_size_gru = 64
        # shared between gru and transformer
        self.num_layers = 1

        # hinrichs Transformer
        self.pos_encoding = 'fixed'
        # TODO possibly in hparam - could be expanded to the other models
        self.activation='gelu'
        self.norm = 'BatchNorm'
        self.freeze = False
        self.n_heads = 8
        self.dim_feedforward = 128

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500

        self.data_augmentation_configs = None


class PHD(object):
    def __init__(self,
                kernel_size=3,
                stride=1,
                dropout=0.5,
                normalize=True,
                shuffle=True,
                drop_last=True,
                mid_channels=64,
                final_out_channels=128,
                features_len=1,
                tcn_layers=[75, 150],
                tcn_kernel_size=17,
                tcn_dropout=0.5,
                lstm_hid=128,
                lstm_n_layers=1,
                lstm_bid=False,
                disc_hid_dim=64,
                DSKN_disc_hid=128,
                hidden_dim=500):
        super(PHD, self).__init__()
        self.sequence_len = 12
        self.scenarios = [("female_hinrichs_dataset", "male_hinrichs_dataset")]
        self.class_names = []
        self.num_classes = 0

        self.num_cont_output_channels = 6

        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # Model configs
        self.input_channels = 6
        self.kernel_size = 32
        self.stride = 6
        self.dropout = 0.1

        self.mid_channels = 64
        # d_model
        self.final_out_channels = 64
        self.features_len = 1

        # SWIFT features
        self.lstm_layers = [256, 16]
        self.bidirectional_lstm = True

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # LTCN features
        self.lctn_layers = [64, 32]
        self.ode_unfolds = 2

        # CfCN features 
        self.cfcn_layers = [64, 32]
        self.backbone_units = 32
        self.backbone_layers = 1
        self.backbone_dropout = 0.1


        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False


        # hinrichs GRU features
        self.hidden_size_gru = 64
        # shared between gru and transformer
        self.num_layers = 1

        # hinrichs Transformer
        self.pos_encoding = 'fixed'
        # TODO possibly in hparam - could be expanded to the other models
        self.activation='gelu'
        self.norm = 'BatchNorm'
        self.freeze = False
        self.n_heads = 8
        self.dim_feedforward = 128

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500

        self.data_augmentation_configs = None