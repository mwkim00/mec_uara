import numpy as np
import torch


class RandAgent():
    name = 'Random'  # Call name without making an instance.

    def __init__(self, config):
        self.name = 'Random'
        self.num_UE = config.num_UE
        self.num_BS = config.num_BS
        self.epsilon = config.compute.epsilon

    def update(self, **kwargs):
        pass

    def decide(self, **kwargs):
        # return np.random.randint(-1, self.num_BS)
        bs = np.random.randint(0, self.num_BS)
        if np.random.rand() < self.epsilon:
            bs = -1

        return bs


class MaxSINRAgent():
    name = 'MaxSINR'  # Call name without making an instance.

    def __init__(self, config):
        self.name = 'MaxSINR'
        self.num_UE = config.num_UE
        self.num_BS = config.num_BS
        self.epsilon = config.compute.epsilon

    def initialize(self):
        pass

    def update(self, **kwargs):
        pass

    def decide(self, **kwargs):
        if np.random.uniform() < self.epsilon:
            return -1
        SNR = kwargs['SNR']
        ue_ind = kwargs['service'].UE_ind

        return np.argmax(SNR[ue_ind, :])


class MaxComputeAgent():
    name = 'MaxCompute'  # Call name without making an instance.

    def __init__(self, config):
        self.name = 'MaxCompute'
        self.num_UE = config.num_UE
        self.num_BS = config.num_BS
        self.epsilon = config.compute.epsilon
        self.sm = torch.nn.Softmax(dim=0)

    def initialize(self):
        pass

    def update(self, **kwargs):
        pass

    def decide(self, **kwargs):
        if np.random.uniform() < self.epsilon:
            return -1

        self.BS = kwargs['BS']
        self.BS_load = np.zeros([self.num_BS])
        for bs_ind, bs in enumerate(self.BS):
            self.BS_load[bs_ind] = bs.flops
            for queue in bs.queue:
                self.BS_load[bs_ind] -= queue.comput_load_remain

        # Add randomness
        prob = self.sm(torch.from_numpy(self.BS_load))
        return torch.multinomial(prob, 1).item()


class CombinedSINRComputeAgent():
    """
    Softmax(Sotmax(SNR) + weight * Softmax(compute load))
    """
    name = 'CombinedSINRCompute'  # Call name without making an instance.

    def __init__(self, config):
        self.name = 'CombinedSINRCompute'
        self.num_UE = config.num_UE
        self.num_BS = config.num_BS
        self.epsilon = config.compute.epsilon
        self.sm = torch.nn.Softmax(dim=0)
        self.weight = 1.0

    def initialize(self):
        pass

    def update(self, **kwargs):
        pass

    def decide(self, **kwargs):
        if np.random.uniform() < self.epsilon:
            return -1

        # SNR part
        SNR = kwargs['SNR']
        ue_ind = kwargs['service'].UE_ind
        cm_info = SNR[ue_ind, :]

        # Compute part
        self.BS = kwargs['BS']
        cp_info = np.zeros([self.num_BS])
        for bs_ind, bs in enumerate(self.BS):
            cp_info[bs_ind] = bs.flops
            for queue in bs.queue:
                cp_info[bs_ind] -= queue.comput_load_remain

        # info = cm_info / sum(cm_info) + self.weight * cp_info / sum(cp_info)
        info = self.sm(torch.from_numpy(cm_info)) + self.weight * self.sm(torch.from_numpy(cp_info))

        # Add randomness
        # prob = self.sm(info)
        prob = info / sum(info)
        return torch.multinomial(prob, 1).item()


class ProposedAgent():
    """
    (4): Proposed
    """
    name = 'Proposed'  # Call name without making an instance.

    def __init__(self, config):
        self.name = 'Proposed'
        self.mu = config.compute.mu
        self.num_UE = config.num_UE
        self.num_BS = config.num_BS
        self.initialize()
        self.record = np.empty([0, 2, self.num_BS])

    def initialize(self):
        #  self.nu = np.random.uniform(size=[2, self.num_BS])
        self.nu = np.zeros([2, self.num_BS]) * 10
        self.d_i = np.zeros([self.num_UE, 1])
        self.f_i = np.zeros([self.num_UE, 1])
        self.P_i = np.zeros([self.num_UE, 1])
        self.F_j = np.zeros([1, self.num_BS]) + 1e-7
        self.R = np.zeros([self.num_UE, self.num_BS]) + 1e-7
        self.X = np.zeros([self.num_UE, self.num_BS])

    def update(self, **kwargs):
        self.t = kwargs['t']
        lr1 = 0.01
        lr2 = 2.0
        gamma = 0.0001
        lr1 = lr1 / (1 + gamma * self.t)
        lr2 = lr2 / (1 + gamma * self.t)
        self.nu[0, :] += lr1 * (-self.nu[0, :] / 2 + np.sum(np.sqrt(self.d_i / self.R) * self.X, axis=0))
        self.nu[1, :] += lr2 * (-self.nu[1, :] / 2 + np.sum(np.sqrt(self.f_i * self.P_i / self.F_j) * self.X, axis=0))
        self.nu[self.nu < 0] = 0.0

        if self.t % 100 == 0:
            self.record = np.append(self.record, np.expand_dims(self.nu, axis=0), axis=0)

    def decide(self, **kwargs):
        # Define variables
        tau = 1  # Temporary variable.
        D_i = np.ones([self.num_UE, 1])  # Temporary variable.
        B_i = np.ones([self.num_UE, 1])  # Temporary variable.
        self.UE = kwargs['UE']
        self.BS = kwargs['BS']
        self.SNR = kwargs['SNR']
        self.BW_per_BS = kwargs['BW_per_BS']
        self.service = kwargs['service']
        self.R = np.log2(1 + self.SNR) * self.BW_per_BS

        for bs_ind, bs in enumerate(self.BS):
            self.F_j[0, bs_ind] = bs.flops

        self.P_i[self.service.UE_ind, :] = 0.99
        self.d_i[self.service.UE_ind, :] = self.service.commun_load
        self.f_i[self.service.UE_ind, :] = self.service.comput_load

        self.F_j = np.zeros([1, self.num_BS])
        for bs_ind, bs in enumerate(self.BS):
            self.F_j[0, bs_ind] = bs.flops

        # First term
        t1 = tau * self.f_i[self.service.UE_ind, :] * (1 - self.P_i[self.service.UE_ind, :]) / self.F_j[0, :]

        # Second term
        C_i = self.service.comput_load_remain * self.P_i[self.service.UE_ind, :] / self.UE[self.service.UE_ind].flops
        t2 = self.mu * D_i[self.service.UE_ind, :] / B_i[self.service.UE_ind, :] + C_i

        # Third term
        t3 = self.nu[0, :] * np.sqrt(self.d_i[self.service.UE_ind, :] / self.R[self.service.UE_ind, :])

        # Fourth term
        t4 = self.nu[1, :] * np.sqrt(
            self.f_i[self.service.UE_ind, :] * self.P_i[self.service.UE_ind, :] / self.F_j[0, :])

        # Local computation decision.
        choice = np.argmin((t1 - t2 + t3 + t4), axis=0)
        min_values = np.min((t1 - t2 + t3 + t4), axis=0)
        #  print(min_values)
        self.X[self.service.UE_ind, :] = 0

        if min_values < 0:
            self.X[self.service.UE_ind, choice] = 1
        else:
            choice = -1

        return choice


class ProposedVisualizeAgent():
    """
    (5): Proposed with visualization of primal and dual problem
    """
    name = 'Proposed - Visualize'  # Call name without making an instance.

    def __init__(self, config):
        self.name = 'Proposed - Visualize'
        self.mu = config.compute.mu
        self.num_UE = config.num_UE
        self.num_BS = config.num_BS
        self.initialize()

        self.period = 1  # Record period. cf. warmup: [0, 100]
        self.start_record = 100

        self.record = np.empty([0, self.num_BS])

        self.a_j_wo_sum = np.zeros([self.num_UE, self.num_BS])
        self.b_j_wo_sum = np.zeros([self.num_UE, self.num_BS])
        self.c_ij = 0
        self.min_values = np.zeros([self.num_UE])

        self.default_lr1 = 0.01
        self.default_lr2 = 2.0
        self.default_gamma = 0.0001

    def _initialize_visualizing_variables(self):
        self.a_j_wo_sum = np.zeros([self.num_UE, self.num_BS])
        self.b_j_wo_sum = np.zeros([self.num_UE, self.num_BS])
        self.c_ij = 0
        # self.min_values = np.zeros([self.num_UE])

    def initialize(self):
        #  self.nu = np.random.uniform(size=[2, self.num_BS])
        self.nu = np.zeros([2, self.num_BS]) * 10
        self.d_i = np.zeros([self.num_UE, 1])
        self.f_i = np.zeros([self.num_UE, 1])
        self.P_i = np.zeros([self.num_UE, 1])
        self.F_j = np.zeros([1, self.num_BS]) + 1e-7
        self.R = np.zeros([self.num_UE, self.num_BS]) + 1e-7
        self.X = np.zeros([self.num_UE, self.num_BS])

    def update(self, **kwargs):
        self.t = kwargs['t']
        lr1 = self.default_lr1 / (1 + self.default_gamma * self.t)
        lr2 = self.default_lr2 / (1 + self.default_gamma * self.t)
        self.nu[0, :] += lr1 * (-self.nu[0, :] / 2 + np.sum(np.sqrt(self.d_i / self.R) * self.X, axis=0))
        self.nu[1, :] += lr2 * (-self.nu[1, :] / 2 + np.sum(np.sqrt(self.f_i * self.P_i / self.F_j) * self.X, axis=0))
        self.nu[self.nu < 0] = 0.0

        if self.t % self.period == 0 and self.t >= self.start_record:  # Prevent recording at t=0
            primal = self._calculate_primal()
            dual = self._calculate_dual()

            record_slice = np.append(self.nu[:, 0], [primal, dual])  # Record BS 1's mu_1, nu_1
            self.record = np.append(self.record, np.expand_dims(record_slice, axis=0), axis=0)
        self._initialize_visualizing_variables()
        # self.c_ij = 0

    def _calculate_primal(self):
        a_j = self.a_j_wo_sum.sum(axis=0) ** 2
        b_j = self.b_j_wo_sum.sum(axis=0) ** 2

        primal = (a_j + b_j).sum() + self.c_ij

        return primal

    def _calculate_dual(self):
        dual = -(self.nu ** 2).sum() / 4 + self.min_values.sum()
        return dual

    def _calculate_terms(self, kwargs):
        """
        Calculate terms for primal and dual problem
        """
        # Define variables
        tau = 1  # Temporary variable.
        D_i = np.ones([self.num_UE, 1])  # Temporary variable.
        B_i = np.ones([self.num_UE, 1])  # Temporary variable.
        self.UE = kwargs['UE']
        self.BS = kwargs['BS']
        self.SNR = kwargs['SNR']
        self.BW_per_BS = kwargs['BW_per_BS']
        self.service = kwargs['service']
        self.R = np.log2(1 + self.SNR) * self.BW_per_BS

        for bs_ind, bs in enumerate(self.BS):
            self.F_j[0, bs_ind] = bs.flops

        self.P_i[self.service.UE_ind, :] = 0.99
        self.d_i[self.service.UE_ind, :] = self.service.commun_load
        self.f_i[self.service.UE_ind, :] = self.service.comput_load

        self.F_j = np.zeros([1, self.num_BS])
        for bs_ind, bs in enumerate(self.BS):
            self.F_j[0, bs_ind] = bs.flops

        # First term
        t1 = tau * self.f_i[self.service.UE_ind, :] * (1 - self.P_i[self.service.UE_ind, :]) / self.F_j[0, :]

        # Second term
        C_i = self.service.comput_load_remain * self.P_i[self.service.UE_ind, :] / self.UE[self.service.UE_ind].flops
        t2 = self.mu * D_i[self.service.UE_ind, :] / B_i[self.service.UE_ind, :] + C_i

        # Third term
        t3 = np.sqrt(self.d_i[self.service.UE_ind, :] / self.R[self.service.UE_ind, :])

        # Fourth term
        t4 = np.sqrt(self.f_i[self.service.UE_ind, :] * self.P_i[self.service.UE_ind, :] / self.F_j[0, :])

        return t1, t2, t3, t4

    def decide(self, **kwargs):
        t1, t2, t3, t4 = self._calculate_terms(kwargs)

        # Local computation decision.
        choice = np.argmin((t1 - t2 + self.nu[0, :] * t3 + self.nu[1, :] * t4), axis=0)
        min_values = np.min((t1 - t2 + self.nu[0, :] * t3 + self.nu[1, :] * t4), axis=0)
        self.X[self.service.UE_ind, :] = 0

        if min_values < 0:
            self.X[self.service.UE_ind, choice] = 1
        else:
            choice = -1

        # Process data for visualization
        if choice != -1:
            self.a_j_wo_sum[self.service.UE_ind, choice] = t3[choice]
            self.b_j_wo_sum[self.service.UE_ind, choice] = t4[choice]
            self.c_ij += (t1 - t2)[choice]
            self.min_values[self.service.UE_ind] = min_values

        return choice
