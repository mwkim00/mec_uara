# Description: Edge computing environment.
from copy import deepcopy
import warnings

import numpy as np


class CommunNet():
    """
    Edge network class.
    """
    def __init__(self, agent, config):
        self.t = 0  # timeslot
        # Load all variables from config
        attr = config.__dict__
        for key, value in attr.items():
            #  print(key, value)
            setattr(self, key, value)
        # Load Agent
        self.agent = agent
        # Setup Queue, Battery, Warmup Time
        for bs in self.BS:
            bs.queue = list()
            bs.energy_consumption = 0.0
        for UE_ind, ue in enumerate(self.UE):
            ue.is_dead = False
            ue.ind = UE_ind
            ue.queue = list()
            ue.busy = False
            ue.remain_battery = np.random.uniform(self.initial_battery[0], self.initial_battery[1]) * ue.battery
            ue.warmup = int(np.random.uniform(self.warmup[0], self.warmup[1]))
        # Setup Task
        self.service_fin = list()
        # Setup Location
        self.initialize_location()
        # Data
        self.data = {
            'latency': [],
            'latency_commun': [],
            'latency_compute_UE': [],
            'latency_compute_BS': [],
            'energy_consumption_local': [],
            'energy_consumption_edge': [],
            # 'energy_remain': np.zeros((self.num_step)),  # Energy change of UEs.
            'energy_consumption': [],
            'service_stats': np.zeros((2)),  # (local, edge)
            # 'RA': np.zeros((self.num_step, self.num_UE, self.num_BS, 2)),  # [RA_commun, RA_comput]
            # 'energy_remain_bottom10': np.zeros((self.num_step)),  # Energy change of UEs with bottom 10% energy
            # 'energy_remain_top10': np.zeros((self.num_step)),  # Energy change of UEs with top 10% energy.
            'energy_dead': np.zeros((self.num_step)),  # Device death time.
        }
        # Helper varables
        init_battery = []
        self.inf_num = 0
        for ue in self.UE:
            init_battery.append(ue.remain_battery / ue.battery)
            if ue.battery == np.inf:
                self.inf_num += 1
        self.top10 = np.argsort(init_battery)[-int(self.num_UE * 0.1) - self.inf_num: -self.inf_num]
        self.bottom10 = np.argsort(init_battery)[:int(self.num_UE * 0.1)]

    def track_results(self):
        for bs in self.BS:
            if not hasattr(bs, 'compute_load'):
                bs.compute_load = list()
            bs.compute_load.append(bs.compute_load)

    def initialize_location(self, ue_option='clustered'):
        """
        Initialize location of UE and BS.
        Compute Channel gain.
        """
        # Random Scatter of BS and UEs
        BS_loc = np.random.uniform(low=-self.channel.cell_size, high=self.channel.cell_size, size=(1, self.num_BS, 2))
        if ue_option == 'original':
            UE_loc = np.random.normal(
                size=(self.num_UE, 1, 2)) * self.channel.cell_size / 3 + self.channel.cell_size * np.random.choice(
                [-0.5, 0.5], size=(self.num_UE, 1, 1))
        elif ue_option == 'clustered':
            # Set cluster info here.
            centers = np.array([[-0.5, -0.5], [1.41, 0.]]) * self.channel.cell_size
            scale = 20

            UE_loc = np.zeros((self.num_UE, 1, 2))
            groups = len(centers) + 1  # The last group is uniform-distributed. The rest are clustered.
            for i in range(groups - 1):
                UE_loc[i * (self.num_UE // groups):(i + 1) * (self.num_UE // groups), :, :] = \
                    np.random.normal(size=(self.num_UE // groups, 1, 2), scale=scale) + centers[i]
            UE_loc[(i + 1) * (self.num_UE // groups):, :, :] = np.random.uniform(
                size=(self.num_UE - (i + 1) * (self.num_UE // groups), 1,
                      2)) * 2 * self.channel.cell_size - self.channel.cell_size
        elif ue_option == 'uniform':
            UE_loc = np.random.uniform(
                size=(self.num_UE, 1, 2)) * 2 * self.channel.cell_size
            UE_loc -= self.channel.cell_size

        # Compute Distance and Channel gain
        d = (((BS_loc - UE_loc) ** 2).sum(axis=2) + 100) ** 0.5  # Add 10^2=100; consider BS height.
        H_dB_org = 41 + 28 * np.log10(d)
        self.C_gain = -H_dB_org + self.channel.SNR_dBm - self.channel.AWGN_dBm \
                      - self.channel.outdoor_wall_dB + self.channel.antenna_gain
        # Randomize channel
        self.H_dB_random = np.random.normal(size=(self.num_UE, self.num_BS)) * self.channel.random_channel_dB
        self.update_channel()

    def update_channel(self):
        # if Prob. 0.1 change channel randomly (soft)
        if np.random.uniform() < 0.1:
            self.H_dB_random = 0.9 * self.H_dB_random + (1 - 0.9 ** 2) ** 0.5 * np.random.normal(
                size=(self.num_UE, self.num_BS)) * self.channel.random_channel_dB
        #  Compute achievable rate
        self.SNR = 10 ** ((self.C_gain + self.H_dB_random) / 10.0)  # SNR.shape = (UE, BS). Change from db scale.
        self.R = np.log2(1 + self.SNR) * self.channel.BW_per_BS
        # Achievable rate. # R.shape = (UE, BS). [bits / s / Hz] x [Hz].

    def get_RA(self):
        """
        Get Resource Allocation table.
        """
        for ue in self.UE:
            ue.RA_comput = ue.flops * self.time_slot
        # Add weights
        weight_commun = np.zeros((self.num_BS))
        weight_comput = np.zeros((self.num_BS))
        for BS_ind, bs in enumerate(self.BS):
            for service in bs.queue:
                if service.commun_load_remain <= 0:
                    weight_comput[BS_ind] += (service.comput_load / (bs.flops * self.time_slot)) ** 0.5
                else:
                    weight_commun[BS_ind] += (service.commun_load / (
                            self.R[service.UE_ind, BS_ind] * self.time_slot)) ** 0.5

        # Obtain RA
        for BS_ind, bs in enumerate(self.BS):
            for service in bs.queue:
                if service.commun_load_remain <= 0:
                    service.RA_commun = 0.0
                    service.RA_comput = bs.flops * self.time_slot * (
                            service.comput_load / (bs.flops * self.time_slot)) ** 0.5 / weight_comput[BS_ind]

                    # self.data['RA'][self.t, service.UE_ind, BS_ind, 1] = service.RA_comput
                else:
                    service.RA_commun = self.R[service.UE_ind, BS_ind] * self.time_slot * (
                            service.commun_load / (self.R[service.UE_ind, BS_ind] * self.time_slot)) ** 0.5 / \
                                        weight_commun[BS_ind]
                    service.RA_comput = 0.0
                    # self.data['RA'][self.t, service.UE_ind, BS_ind, 0] = service.RA_commun

            # self.data['RA'][self.t] = np.array([[[service.RA_commun, service.RA_comput] for service in bs.queue] for bs in self.BS])

    def proceed(self):
        """
        Proceed to the next time step with the UA scheme obtained from Agent.
        """
        self.update_channel()
        # Allocate Tasks
        for ue in self.UE:
            ue.energy_consump_comput = 0.0
            ue.energy_consump_commun = 0.0
            # if ue.warmup <= self.t and ue.busy == False:
            if ue.warmup <= self.t and ue.busy == False and not ue.is_dead:
                # Generate Service
                service_ = np.random.choice(list(self.service.keys()), p=list(self.service.values()))
                service_ = deepcopy(self.compute.service_list[service_])
                service_.comput_load_remain = service_.comput_load
                service_.generated_time = self.t
                service_.UE_ind = ue.ind
                service_.host = self.agent.decide(
                    UE=self.UE,
                    BS=self.BS,
                    service=service_,
                    SNR=self.SNR,
                    BW_per_BS=self.channel.BW_per_BS)
                if service_.host == -1:
                    self.data['service_stats'][0] += 1
                    service_.transmitted_time = self.t
                    ue.queue.append(service_)
                else:
                    self.data['service_stats'][1] += 1
                    service_.commun_load_remain = service_.commun_load
                    self.BS[service_.host].queue.append(service_)
                ue.busy = True
        self.get_RA()
        self.agent.update(t=self.t)
        # Process Computation (in UE)
        for ue in self.UE:
            for service in ue.queue[:]:
                if service.comput_load_remain <= ue.RA_comput:
                    ue.RA_comput = min(service.comput_load_remain, ue.RA_comput)
                    service.comput_load_remain = 0
                    service.finished_time = self.t
                    self.service_fin.append(service)
                    ue.queue.remove(service)
                    ue.busy = False
                    self.data['latency'].append(service.finished_time - service.generated_time)
                    self.data['latency_compute_UE'].append(service.finished_time - service.generated_time)
                else:
                    service.comput_load_remain -= ue.RA_comput
                ue.energy_consump_comput = ue.RA_comput / self.compute.flops_per_watt / 3600  # flops / (flops / watt / sec) / (3600 sec / hour)
        # Process Communication & Computation (in BS)
        for BS_ind, bs in enumerate(self.BS):
            if len(bs.queue) > 0:
                for service in bs.queue[:]:
                    # Computing and Energy Consumption
                    if service.commun_load_remain <= 0:
                        bs.energy_consumption += service.RA_comput / self.compute.flops_per_watt / 3600
                        self.data['energy_consumption_edge'].append(bs.energy_consumption)
                        self.data['energy_consumption'].append(bs.energy_consumption)
                        if service.comput_load_remain <= service.RA_comput:
                            service.comput_load_remain = 0
                            self.UE[service.UE_ind].busy = False
                            service.finished_time = self.t
                            self.service_fin.append(service)
                            bs.queue.remove(service)
                            self.data['latency'].append(service.finished_time - service.generated_time)
                            self.data['latency_compute_BS'].append(service.finished_time - service.transmitted_time)
                        else:
                            service.comput_load_remain -= service.RA_comput
                    # Communication and Energy Consumption
                    else:
                        self.UE[service.UE_ind].energy_consump_commun = service.RA_commun / (self.R[
                                                                                                 service.UE_ind, BS_ind] * self.time_slot) * self.channel.SNR_w * self.time_slot
                        if service.commun_load_remain <= service.RA_commun:
                            service.commun_load_remain = 0
                            service.transmitted_time = self.t
                            self.data['latency_commun'].append(service.transmitted_time - service.generated_time)
                        else:
                            service.commun_load_remain -= service.RA_commun

        terminate = True
        for ue in self.UE:
            ue.remain_battery -= ue.energy_consump_comput
            ue.remain_battery -= ue.energy_consump_commun

            if not ue.remain_battery == np.inf and not ue.is_dead:
                terminate = False

            # Collect data
            if ue.remain_battery <= 0.0 and not ue.is_dead:
                ue.is_dead = True
                self.data['energy_dead'][self.t] += 1
            # if ue.ind in self.top10:
            #     self.data['energy_remain_top10'][self.t] += ue.remain_battery / ue.battery / len(self.top10)
            # if ue.ind in self.bottom10:
            #     self.data['energy_remain_bottom10'][self.t] += ue.remain_battery / ue.battery / len(self.bottom10)
            if ue.battery != np.inf:
                self.data['energy_consumption_local'].append(ue.energy_consump_commun + ue.energy_consump_comput)
            #     self.data['energy_remain'][self.t] += ue.remain_battery / ue.battery / (self.num_UE - self.inf_num)
            self.data['energy_consumption'].append(ue.energy_consump_commun + ue.energy_consump_comput)
        self.t += 1
        return terminate

    def summary(self):
        """
        Print summary of the simulation
        """
        warnings.filterwarnings('error')  # Catch warnings same as errors.

        Avg_delay = list()
        Avg_delay_commun = list()
        Avg_delay_comput = list()
        Avg_delay_UE = list()
        Avg_delay_BS = list()
        for service in self.service_fin:
            Avg_delay.append(service.finished_time - service.generated_time)
            Avg_delay_commun.append(service.transmitted_time - service.generated_time)
            Avg_delay_comput.append(service.finished_time - service.transmitted_time)
            if service.host == -1:
                Avg_delay_UE.append(service.finished_time - service.generated_time)
            else:
                Avg_delay_BS.append(service.finished_time - service.generated_time)
            #  print('service_generated_time: ', service.generated_time)
            #  print('service_finished_time: ', service.finished_time)
        print("Average Delay: ", np.mean(Avg_delay))
        print("Average Delay (Communication): ", np.mean(Avg_delay_commun))
        print("Average Delay (Computation): ", np.mean(Avg_delay_comput))
        try:
            print("Average Delay (UE): ", np.mean(Avg_delay_UE))
        except RuntimeWarning:
            print("Average Delay (UE): No service finished in UE.")
        try:
            print("Average Delay (BS): ", np.mean(Avg_delay_BS))
        except RuntimeWarning:
            print("Average Delay (BS): No service finished in BS")
        print(f"Total Energy consumption: {np.mean(self.data['energy_consumption'])} [Wh]")
        print(f"Energy consumption (local): {np.mean(self.data['energy_consumption_local'])} [Wh]")
        try:
            print(f"Energy consumption (edge): {np.mean(self.data['energy_consumption_edge'])} [Wh]")
        except RuntimeWarning:
            print(f"Energy consumption (edge): No service finished in BS")


class CommunNetVisualize():
    """
    Edge network class, used to visualize the effects of different learning rates. See main_visualize.py
    """
    def __init__(self, agent, config):
        self.t = 0  # timeslot
        # Load all variables from config
        attr = config.__dict__
        for key, value in attr.items():
            #  print(key, value)
            setattr(self, key, value)
        # Load Agent
        self.agent = agent
        # Setup Queue, Battery, Warmup Time
        for bs in self.BS:
            bs.queue = list()
            bs.energy_consumption = 0.0
        for UE_ind, ue in enumerate(self.UE):
            ue.is_dead = False
            ue.ind = UE_ind
            ue.queue = list()
            ue.busy = False
            ue.remain_battery = np.random.uniform(self.initial_battery[0], self.initial_battery[1]) * ue.battery
            ue.warmup = int(np.random.uniform(self.warmup[0], self.warmup[1]))
        # Setup Task
        self.service_fin = list()
        # Setup Location
        self.initialize_location()
        # Data
        self.data = {
            'latency': [],
            'latency_commun': [],
            'latency_compute_UE': [],
            'latency_compute_BS': [],
            'energy_consumption_local': [],
            'energy_consumption_edge': [],
            'energy_consumption': [],
            'service_stats': np.zeros((2)),  # (local, edge)
            'energy_dead': np.zeros((self.num_step)),  # Device death time.
        }
        # Helper varables
        init_battery = []
        self.inf_num = 0
        for ue in self.UE:
            init_battery.append(ue.remain_battery / ue.battery)
            if ue.battery == np.inf:
                self.inf_num += 1
        self.top10 = np.argsort(init_battery)[-int(self.num_UE * 0.1) - self.inf_num: -self.inf_num]
        self.bottom10 = np.argsort(init_battery)[:int(self.num_UE * 0.1)]

    def track_results(self):
        for bs in self.BS:
            if not hasattr(bs, 'compute_load'):
                bs.compute_load = list()
            bs.compute_load.append(bs.compute_load)

    def initialize_location(self, ue_option='clustered'):
        """
        Initialize location of UE and BS
        Compute Channel gain
        """
        # Random Scatter of BS and UEs
        BS_loc = np.random.uniform(low=-self.channel.cell_size, high=self.channel.cell_size, size=(1, self.num_BS, 2))
        if ue_option == 'original':
            UE_loc = np.random.normal(
                size=(self.num_UE, 1, 2)) * self.channel.cell_size / 3 + self.channel.cell_size * np.random.choice(
                [-0.5, 0.5], size=(self.num_UE, 1, 1))
        elif ue_option == 'clustered':
            # Set cluster info here.
            centers = np.array([[-0.5, -0.5], [1.41, 0.]]) * self.channel.cell_size
            scale = 20

            UE_loc = np.zeros((self.num_UE, 1, 2))
            groups = len(centers) + 1  # The last group is uniform-distributed. The rest are clustered.
            for i in range(groups - 1):
                UE_loc[i * (self.num_UE // groups):(i + 1) * (self.num_UE // groups), :, :] = \
                    np.random.normal(size=(self.num_UE // groups, 1, 2), scale=scale) + centers[i]
            UE_loc[(i + 1) * (self.num_UE // groups):, :, :] = np.random.uniform(
                size=(self.num_UE - (i + 1) * (self.num_UE // groups), 1,
                      2)) * 2 * self.channel.cell_size - self.channel.cell_size
        elif ue_option == 'uniform':
            UE_loc = np.random.uniform(
                size=(self.num_UE, 1, 2)) * 2 * self.channel.cell_size
            UE_loc -= self.channel.cell_size

        # Compute Distance and Channel gain
        d = (((BS_loc - UE_loc) ** 2).sum(axis=2) + 100) ** 0.5  # Add 10^2=100; consider BS height.
        H_dB_org = 41 + 28 * np.log10(d)
        self.C_gain = -H_dB_org + self.channel.SNR_dBm - self.channel.AWGN_dBm \
                      - self.channel.outdoor_wall_dB + self.channel.antenna_gain
        # Randomize channel
        self.H_dB_random = np.random.normal(size=(self.num_UE, self.num_BS)) * self.channel.random_channel_dB
        self.update_channel()

    def update_channel(self):
        # if Prob. 0.1 change channel randomly (soft)
        if np.random.uniform() < 0.1:
            self.H_dB_random = 0.9 * self.H_dB_random + (1 - 0.9 ** 2) ** 0.5 * np.random.normal(
                size=(self.num_UE, self.num_BS)) * self.channel.random_channel_dB
        #  Compute achievable rate
        self.SNR = 10 ** ((self.C_gain + self.H_dB_random) / 10.0)  # SNR.shape = (UE, BS). Change from db scale.
        self.R = np.log2(1 + self.SNR) * self.channel.BW_per_BS
        # Achievable rate. # R.shape = (UE, BS). [bits / s / Hz] x [Hz].

    def get_RA(self):
        """
        Get Resource Allocation table
        """
        for ue in self.UE:
            ue.RA_comput = ue.flops * self.time_slot
        # Add weights
        weight_commun = np.zeros((self.num_BS))
        weight_comput = np.zeros((self.num_BS))
        for BS_ind, bs in enumerate(self.BS):
            for service in bs.queue:
                if service.commun_load_remain <= 0:
                    weight_comput[BS_ind] += (service.comput_load / (bs.flops * self.time_slot)) ** 0.5
                else:
                    weight_commun[BS_ind] += (service.commun_load / (
                            self.R[service.UE_ind, BS_ind] * self.time_slot)) ** 0.5

        # Obtain RA
        for BS_ind, bs in enumerate(self.BS):
            for service in bs.queue:
                if service.commun_load_remain <= 0:
                    service.RA_commun = 0.0
                    service.RA_comput = bs.flops * self.time_slot * (
                            service.comput_load / (bs.flops * self.time_slot)) ** 0.5 / weight_comput[BS_ind]

                    # self.data['RA'][self.t, service.UE_ind, BS_ind, 1] = service.RA_comput
                else:
                    service.RA_commun = self.R[service.UE_ind, BS_ind] * self.time_slot * (
                            service.commun_load / (self.R[service.UE_ind, BS_ind] * self.time_slot)) ** 0.5 / \
                                        weight_commun[BS_ind]
                    service.RA_comput = 0.0
                    # self.data['RA'][self.t, service.UE_ind, BS_ind, 0] = service.RA_commun

    def proceed(self):
        """
        Proceed to the next time step with the UA scheme obtained from Agent.
        """
        # """
        # Use the same state (service requirements, environment status) for smoother convergence graph
        # Slows the code down
        # THE CODE INCREASES THE DUALITY GAP
        if self.t == self.agent.start_record:
            self.UE_copy = deepcopy(self.UE)
            self.BS_copy = deepcopy(self.BS)

        if self.t > self.agent.start_record:
            self.UE = deepcopy(self.UE_copy)
            self.BS = deepcopy(self.BS_copy)
        # """
        self.update_channel()

        # Allocate Tasks
        for i, ue in enumerate(self.UE):
            ue.energy_consump_comput = 0.0
            ue.energy_consump_commun = 0.0
            if ue.warmup <= self.t and not ue.is_dead:  # Always create tasks to better visualize the convergence graph
                # Generate Service
                np.random.seed(i)  # Fix seed for better convergence graph, but fixing the seed messes up the latency
                service_ = np.random.choice(list(self.service.keys()), p=list(self.service.values()))
                service_ = deepcopy(self.compute.service_list[service_])
                service_.comput_load_remain = service_.comput_load
                service_.generated_time = self.t
                service_.UE_ind = ue.ind
                service_.host = self.agent.decide(
                    UE=self.UE,
                    BS=self.BS,
                    service=service_,
                    SNR=self.SNR,
                    BW_per_BS=self.channel.BW_per_BS)
                if service_.host == -1:
                    self.data['service_stats'][0] += 1
                    service_.transmitted_time = self.t
                    ue.queue = []
                    ue.queue.append(service_)
                else:
                    self.data['service_stats'][1] += 1
                    service_.commun_load_remain = service_.commun_load
                    self.BS[service_.host].queue.append(service_)
                ue.busy = True
        self.get_RA()
        self.agent.update(t=self.t)
        # Process Computation (in UE)
        for ue in self.UE:
            for service in ue.queue[:]:
                if service.comput_load_remain <= ue.RA_comput:
                    ue.RA_comput = min(service.comput_load_remain, ue.RA_comput)
                    service.comput_load_remain = 0
                    service.finished_time = self.t
                    self.service_fin.append(service)
                    ue.queue.remove(service)
                    ue.busy = False
                    self.data['latency'].append(service.finished_time - service.generated_time)
                    self.data['latency_compute_UE'].append(service.finished_time - service.generated_time)
                else:
                    service.comput_load_remain -= ue.RA_comput
                ue.energy_consump_comput = ue.RA_comput / self.compute.flops_per_watt / 3600  # flops / (flops / watt / sec) / (3600 sec / hour)
        # Process Communication & Computation (in BS)
        for BS_ind, bs in enumerate(self.BS):
            if len(bs.queue) > 0:
                for service in bs.queue[:]:
                    # Computing and Energy Consumption
                    if service.commun_load_remain <= 0:
                        bs.energy_consumption += service.RA_comput / self.compute.flops_per_watt / 3600
                        self.data['energy_consumption_edge'].append(bs.energy_consumption)
                        self.data['energy_consumption'].append(bs.energy_consumption)
                        if service.comput_load_remain <= service.RA_comput:
                            service.comput_load_remain = 0
                            self.UE[service.UE_ind].busy = False
                            service.finished_time = self.t
                            self.service_fin.append(service)
                            bs.queue.remove(service)
                            self.data['latency'].append(service.finished_time - service.generated_time)
                            self.data['latency_compute_BS'].append(service.finished_time - service.transmitted_time)
                        else:
                            service.comput_load_remain -= service.RA_comput
                    # Communication and Energy Consumption
                    else:
                        self.UE[service.UE_ind].energy_consump_commun = service.RA_commun / (self.R[
                                                                                                 service.UE_ind, BS_ind] * self.time_slot) * self.channel.SNR_w * self.time_slot
                        if service.commun_load_remain <= service.RA_commun:
                            service.commun_load_remain = 0
                            service.transmitted_time = self.t
                            self.data['latency_commun'].append(service.transmitted_time - service.generated_time)
                        else:
                            service.commun_load_remain -= service.RA_commun

        terminate = True
        for ue in self.UE:
            ue.remain_battery -= ue.energy_consump_comput
            ue.remain_battery -= ue.energy_consump_commun

            if not ue.remain_battery == np.inf and not ue.is_dead:
                terminate = False

            # Collect data
            if ue.remain_battery <= 0.0 and not ue.is_dead:
                ue.is_dead = True
                self.data['energy_dead'][self.t] += 1
            # if ue.ind in self.top10:
            #     self.data['energy_remain_top10'][self.t] += ue.remain_battery / ue.battery / len(self.top10)
            # if ue.ind in self.bottom10:
            #     self.data['energy_remain_bottom10'][self.t] += ue.remain_battery / ue.battery / len(self.bottom10)
            if ue.battery != np.inf:
                self.data['energy_consumption_local'].append(ue.energy_consump_commun + ue.energy_consump_comput)
            #     self.data['energy_remain'][self.t] += ue.remain_battery / ue.battery / (self.num_UE - self.inf_num)
            self.data['energy_consumption'].append(ue.energy_consump_commun + ue.energy_consump_comput)
        self.t += 1
        return terminate

    def summary(self):
        """
        Print summary of the simulation
        """
        warnings.filterwarnings('error')  # Catch warnings same as errors.

        Avg_delay = list()
        Avg_delay_commun = list()
        Avg_delay_comput = list()
        Avg_delay_UE = list()
        Avg_delay_BS = list()
        for service in self.service_fin:
            Avg_delay.append(service.finished_time - service.generated_time)
            Avg_delay_commun.append(service.transmitted_time - service.generated_time)
            Avg_delay_comput.append(service.finished_time - service.transmitted_time)
            if service.host == -1:
                Avg_delay_UE.append(service.finished_time - service.generated_time)
            else:
                Avg_delay_BS.append(service.finished_time - service.generated_time)
            #  print('service_generated_time: ', service.generated_time)
            #  print('service_finished_time: ', service.finished_time)
        print("Average Delay: ", np.mean(Avg_delay))
        print("Average Delay (Communication): ", np.mean(Avg_delay_commun))
        print("Average Delay (Computation): ", np.mean(Avg_delay_comput))
        try:
            print("Average Delay (UE): ", np.mean(Avg_delay_UE))
        except RuntimeWarning:
            print("Average Delay (UE): No service finished in UE.")
        try:
            print("Average Delay (BS): ", np.mean(Avg_delay_BS))
        except RuntimeWarning:
            print("Average Delay (BS): No service finished in BS")
        print(f"Total Energy consumption: {np.mean(self.data['energy_consumption'])} [Wh]")
        print(f"Energy consumption (local): {np.mean(self.data['energy_consumption_local'])} [Wh]")
        try:
            print(f"Energy consumption (edge): {np.mean(self.data['energy_consumption_edge'])} [Wh]")
        except RuntimeWarning:
            print(f"Energy consumption (edge): No service finished in BS")
