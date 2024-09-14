import math
from copy import deepcopy
import warnings

import numpy as np
from easydict import EasyDict


class Channel:
    def __init__(self):
        """
        Channel parameters.
        """
        self.cell_size = 200  # meter
        self.SNR_dBm = 30
        self.SNR_w = 10 ** (self.SNR_dBm / 10) * 1e-3  # W
        self.outdoor_wall_dB = 10
        self.antenna_gain = 0  # consider antenna gain
        self.random_channel_dB = 5
        self.BW_per_BS = 10e6  # 10MHz
        self.AWGN_dBm = -174 + 10 * np.log10(self.BW_per_BS)  # Log scale, add (multiply) frequency to change W/Hz to W.


class Compute:
    def __init__(self):
        """
        Computational parameters.
        """

        self.mu = 1.0  # Unit conversion parameter. [sec / ratio] Battery loss function when computing at local.
        self.delta = 2.6  # Energy consumption parameter. See alpha-fairness paper page 4 & Table 1.
        self.flops_per_watt = 10e9  # [flops / sec / watt] 1 GFLOPS = 1e9 FLOPs.

        # Services: [Llama, MMSeg]
        self.ResNet50 = EasyDict({
            'commun_load': 6.0e6,  # [bit], 12 MB
            'comput_load': 4.2e9,
            'compute_load_vram': 2.44e8,  # [byte]
        })

        self.ResNet18 = EasyDict({
            'commun_load': 6.0e6,  # [bit], 12 MB
            'comput_load': 1.8e9,  # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
            'compute_load_vram': 4.47e7,  # [byte]
        })

        self.MobileNet_v2 = EasyDict({
            'commun_load': 3.2e7,  # [bit], 4 MB
            'comput_load': 3.0e8,
            'compute_load_vram': 1.4e7,  # [byte]
        })

        self.Llama = EasyDict({  # 7 billion parameters model.
            'commun_load': 4096,  # [bit], 512 x 8 bits
            'comput_load': 5.0e13,  # [flops], 9.74 [sec] x 35.58e12 [FLOPS, RTX 3090] (18.34 - 8.6 [sec] for loading.)
            'compute_load_vram': 1.515e10,  # [byte]
        })

        self.MMSeg_san = EasyDict({
            'commun_load': 9.6e7,  # [bit], 12 MB
            'comput_load': 7.23e13,  # [flops], 4 [sec] x 35.58e12 [FLOPS, RTX 3090] (Loading time not included.)
            'compute_load_vram': 3.51e9,  # [byte]
        })

        self.MMseg_pspnet = EasyDict({
            'commun_load': 3.2e7,  # [bit], 4 MB
            'comput_load': 5.2e13,  # [flops], 2.87 [sec] x 35.58e12 [FLOPS, RTX 3090] (Loading time not included.)
            'compute_load_vram': 5.12e9,  # [byte]
        })

        self.MMseg_mobile_v3 = EasyDict({
            'commun_load': 3.2e7,  # [bit], 4 MB
            'comput_load': 8e12,  # [flops], 2.64 [sec] x 35.58e12 [FLOPS, RTX 3090] (Loading time not included.)
            'compute_load_vram': 7.42e9,  # [byte]
        })

        self.service_list = dict({
            'ResNet50': self.ResNet50,
            'ResNet18': self.ResNet18,
            'MobileNet_v2': self.MobileNet_v2,
            'Llama': self.Llama,
            'MMSeg_san': self.MMSeg_san,
            'MMSeg_pspnet': self.MMseg_pspnet,
            'MMSeg_mobile_v3': self.MMseg_mobile_v3,
        })

        # Devices: [Galaxy_S23, iPhone_14, Raspberry_Pi_4, Apple_M1, RTX_2080, RTX_3090]
        self.Galaxy_S23 = EasyDict({
            'name': 'Galaxy_S23',
            'flops': 3.681e12,  # [flops / sec]
            'vram': 8,  # [GB]
            # 'threads': 8, # [#num]
            'idle': 0.00001,  # [W]
            'battery': 15.1,  # [Wh]
            'is_dead': False,
        })

        self.iPhone_14 = EasyDict({
            'name': 'iPhone_14',
            'flops': 2e12,  # [flops / sec]
            'vram': 6,  # [GB]
            # 'threads': 6, # [#num]
            'idle': 0.00001,  # [W]
            'battery': 12.7,  # [Wh]
            'is_dead': False,
        })

        self.Mate_60 = EasyDict({
            'name': 'Mate_60',
            'flops': 2.06e12,  # [flops / sec]
            'vram': 6,  # [GB]
            # 'threads': 6, # [#num]
            'idle': 0.00001,  # [W]
            'battery': 18.4,  # [Wh]
            'is_dead': False,
        })

        self.Raspberry_Pi_4 = EasyDict({
            'name': 'Raspberry_Pi_4',
            'flops': 24e9,  # [flops / sec]
            'vram': 4,  # [GB]
            # 'threads': 4, # [#num]
            'idle': 0.00001,  # [W]
            'battery': math.inf,  # [Wh]
            'is_dead': False,
        })

        self.Apple_M1 = EasyDict({
            'name': 'Apple_M1',
            'flops': 2.6e12,  # [flops / sec]
            'vram': 16,  # [GB]
            # 'threads': 8, # [#num]
            'idle': 0.00001,  # [W]
            'battery': math.inf,  # [Wh]
            'is_dead': False,
        })

        self.RTX_2080 = EasyDict({
            'name': 'RTX_2080',
            'flops': 11.2e12,  # [flops / sec]
            'vram': 8,  # [GB]
            # 'threads': 4352, # [#num]
            'idle': 0.00001,  # [W]
            'battery': math.inf,  # [Wh]
            'is_dead': False,
        })

        self.RTX_3090 = EasyDict({
            'name': 'RTX_3090',
            'flops': 35.6e12,  # [flops / sec]
            'vram': 24,  # [GB]
            # 'threads': 10496, # [#num]
            'idle': 0.00001,  # [W]
            'battery': math.inf,  # [Wh]
            'is_dead': False,
        })

        self.A6000 = EasyDict({
            'name': 'A6000',
            'flops': 38.7e12,  # [flops / sec]
            'vram': 48,  # [GB]
            # 'threads': 10496, # [#num]
            'idle': 0.00001,  # [W]
            'battery': math.inf,  # [Wh]
            'is_dead': False,
        })

        self.RTX_4080 = EasyDict({
            'name': 'RTX_4080',
            'flops': 48.7e12,  # [flops / sec]
            'vram': 16,  # [GB]
            # 'threads': 10496, # [#num]
            'idle': 0.00001,  # [W]
            'battery': math.inf,  # [Wh]
            'is_dead': False,
        })

        self.RTX_4090 = EasyDict({
            'name': 'RTX_4090',
            'flops': 82.9e12,  # [flops / sec]
            'vram': 24,  # [GB]
            # 'threads': 10496, # [#num]
            'idle': 0.00001,  # [W]
            'battery': math.inf,  # [Wh]
            'is_dead': False,
        })

        self.device_list = dict({
            'Galaxy_S23': self.Galaxy_S23,
            'iPhone_14': self.iPhone_14,
            'Mate_60': self.Mate_60,
            'Raspberry_Pi_4': self.Raspberry_Pi_4,
            'Apple_M1': self.Apple_M1,
            'RTX_2080': self.RTX_2080,
            'RTX_3090': self.RTX_3090,
            'A6000': self.A6000,
            'RTX_4080': self.RTX_4080,
            'RTX_4090': self.RTX_4090,
        })


class BASE:
    """
    Hyperparameters.
    """

    def __init__(self, ue=20, bs=4, mu=1, epsilon=0.0):
        self.compute = Compute()
        self.compute.mu = mu
        self.compute.epsilon = epsilon
        self.channel = Channel()

        self.num_iter = 100
        self.num_step = 10000
        self.time_slot = 0.1  # [sec]
        self.P_i = 0.99  # 99% of the task is affected by multi-core computation.

        # Set User devices 
        self.UE = np.random.choice(
            ['Galaxy_S23', 'iPhone_14', 'Mate_60', 'Apple_M1'],
            size=ue,
            p=[0.25, 0.25, 0.25, 0.25]
        )
        if bs == 4:
            self.BS = ['RTX_3090'] * 1 + ['A6000'] * 1 + ['RTX_4080'] * 1 + ['RTX_4090'] * 1
        elif bs == 8:
            self.BS = ['RTX_2090'] * 1 + ['RTX_3090'] * 1 + ['A6000'] * 2 + ['RTX_4080'] * 2 + ['RTX_4090'] * 2
        elif bs == 12:
            self.BS = ['RTX_2090'] * 2 + ['RTX_3090'] * 2 + ['A6000'] * 2 + ['RTX_4080'] * 3 + ['RTX_4090'] * 3
        else:
            warnings.warn('Unexpected number of BSs.')
            self.BS = np.random.choice(
                ['RTX_2080', 'RTX_3090', 'A6000'],
                size=bs
            )

        self.service = dict({
            'ResNet50': 0.2,
            'ResNet18': 0.1,
            'MobileNet_v2': 0.1,
            'Llama': 0.1,
            'MMSeg_san': 0.2,
            'MMSeg_pspnet': 0.2,
            'MMSeg_mobile_v3': 0.1,
        })

        self.__update__()

    def __update__(self):
        self.UE = [deepcopy(self.compute.device_list[device]) for device in self.UE]
        self.BS = [deepcopy(self.compute.device_list[device]) for device in self.BS]
        self.num_BS = len(self.BS)
        self.num_UE = len(self.UE)


class highcompute(BASE):
    def __init__(self, ue=20, bs=4, mu=1, epsilon=0.0):
        super().__init__()
        # Customizing should be added below.
        self.P_i = 0.99  # 99% of the task is affected by multi-core computation.

        # Unnecessary
        self.compute.mu = mu
        self.compute.epsilon = epsilon

        # Set User devices
        self.UE = np.random.choice(
            ['Galaxy_S23', 'iPhone_14', 'Mate_60', 'Apple_M1'],
            size=ue,
            p=[0.25, 0.25, 0.25, 0.25]
        )
        if bs == 4:
            self.BS = ['RTX_2080'] * 1 + ['RTX_3090'] * 1 + ['A6000'] * 2
        elif bs == 8:
            self.BS = ['RTX_2080'] * 2 + ['RTX_3090'] * 3 + ['A6000'] * 3
        elif bs == 12:
            self.BS = ['RTX_2080'] * 4 + ['RTX_3090'] * 4 + ['A6000'] * 4
        else:
            warnings.warn('Unexpected number of BSs.')
            self.BS = np.random.choice(
                ['RTX_2080', 'RTX_3090', 'A6000'],
                size=bs
            )

        self.service = dict({
            'ResNet50': 0.1,
            'ResNet18': 0.1,
            'MobileNet_v2': 0.025,
            'Llama': 0.7,
            'MMSeg_san': 0.025,
            'MMSeg_pspnet': 0.025,
            'MMSeg_mobile_v3': 0.025,
        })

        self.num_iter = 100
        self.num_step = 10000

        self.initial_battery = [0.6, 1.0]  # [min, max]
        self.warmup = [0, 100]  # [min, max]
        self.__update__()


class highcommun(BASE):
    def __init__(self, ue=20, bs=4, mu=1, epsilon=0.0):
        super().__init__()
        # Customizing should be added below.
        self.P_i = 0.99  # 99% of the task is affected by multi-core computation.

        # Unnecessary
        self.compute.mu = mu
        self.compute.epsilon = epsilon

        # Set User devices
        self.UE = np.random.choice(
            ['Galaxy_S23', 'iPhone_14', 'Mate_60', 'Apple_M1'],
            size=ue,
            p=[0.25, 0.25, 0.25, 0.25]
        )
        if bs == 4:
            self.BS = ['RTX_2080'] * 1 + ['RTX_3090'] * 1 + ['A6000'] * 2
        elif bs == 8:
            self.BS = ['RTX_2080'] * 2 + ['RTX_3090'] * 3 + ['A6000'] * 3
        elif bs == 12:
            self.BS = ['RTX_2080'] * 4 + ['RTX_3090'] * 4 + ['A6000'] * 4
        else:
            warnings.warn('Unexpected number of BSs.')
            self.BS = np.random.choice(
                ['RTX_2080', 'RTX_3090', 'A6000'],
                size=bs
            )

        self.service = dict({
            'ResNet50': 0.1,
            'ResNet18': 0.1,
            'MobileNet_v2': 0.7,
            'Llama': 0.025,
            'MMSeg_san': 0.025,
            'MMSeg_pspnet': 0.025,
            'MMSeg_mobile_v3': 0.025,
        })

        self.num_iter = 100
        self.num_step = 10000

        self.initial_battery = [0.6, 1.0]  # [min, max]
        self.warmup = [0, 100]  # [min, max]
        self.__update__()


class base(BASE):
    def __init__(self, ue=20, bs=4, mu=1, epsilon=0.0):
        super().__init__()
        # Customizing should be added below.
        self.P_i = 0.99  # 99% of the task is affected by multi-core computation.

        self.compute.mu = mu
        self.compute.epsilon = epsilon

        # Set User devices
        self.UE = np.random.choice(
            ['Galaxy_S23', 'iPhone_14', 'Mate_60', 'Apple_M1'],
            size=ue,
            p=[0.25, 0.25, 0.25, 0.25]
        )
        if bs == 4:
            self.BS = ['RTX_2080'] * 1 + ['RTX_3090'] * 1 + ['A6000'] * 2
        elif bs == 8:
            self.BS = ['RTX_2080'] * 2 + ['RTX_3090'] * 3 + ['A6000'] * 3
        elif bs == 12:
            self.BS = ['RTX_2080'] * 4 + ['RTX_3090'] * 4 + ['A6000'] * 4
        else:
            warnings.warn('Unexpected number of BSs.')
            self.BS = np.random.choice(
                ['RTX_2080', 'RTX_3090', 'A6000'],
                size=bs
            )

        self.service = dict({
            'ResNet50': 0.2,
            'ResNet18': 0.1,
            'MobileNet_v2': 0.1,
            'Llama': 0.1,
            'MMSeg_san': 0.2,
            'MMSeg_pspnet': 0.2,
            'MMSeg_mobile_v3': 0.1,
        })

        self.num_iter = 100
        self.num_step = 10000

        self.initial_battery = [0.6, 1.0]  # [min, max]
        self.warmup = [0, 100]  # [min, max]
        self.__update__()


class highcompute_10(highcompute):
    def __init__(self, ue=20, bs=4, mu=1):
        super().__init__(ue=ue, bs=bs, mu=mu)
        self.num_iter = 10


class highcommun_10(highcommun):
    def __init__(self, ue=20, bs=4, mu=1):
        super().__init__(ue=ue, bs=bs, mu=mu)
        self.num_iter = 10


class base_10(base):
    def __init__(self, ue=20, bs=4, mu=1):
        super().__init__(ue=ue, bs=bs, mu=mu)
        self.num_iter = 10


class test(BASE):
    def __init__(self, ue=20, bs=4, mu=1):
        super().__init__()
        print(f'\n\nTest config.\n'
              f'UE: {ue}, BS: {bs}, mu: {mu}.')

        self.compute.mu = mu
        self.BS = np.random.choice(
            ['RTX_2080', 'RTX_3090', 'A6000'],
            size=bs
        )
        self.UE = np.random.choice(
            ['Galaxy_S23', 'iPhone_14', 'Apple_M1'],
            size=ue,
            p=[1.0, 0.0, 0.0]
        )
        self.service = dict({
            'Llama': 1.0,
            'MMSeg_san': 0.0,
            'MMSeg_pspnet': 0.0,
            'MMSeg_mobile_v3': 0.0,
        })

        self.num_iter = 1
        self.num_step = 10000

        self.initial_battery = [0.6, 1.0]  # [min, max]
        self.warmup = [0, 100]  # [min, max]
        self.__update__()
