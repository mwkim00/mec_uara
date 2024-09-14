"""
A network emulator for the paper "Distributed Task Offloading and Resource Allocation for Latency Minimization in Mobile Edge Computing Networks"


"""
import argparse
import os
import pprint

from tqdm import tqdm
import scipy.io as io

from agent import *
from configs import get_config
from env import CommunNet
from utils import set_seed, process_data, plot_agent_data


def configuration():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',
                        type=str,
                        default='base'
                        )
    parser.add_argument('--agent_name', '-a',
                        type=str,
                        default='proposed'
                        )
    parser.add_argument('--ue', '-u',
                        type=int,
                        default=80
                        )
    parser.add_argument('--bs', '-b',
                        type=int,
                        default=4
                        )
    parser.add_argument('--mu', '-m',
                        type=float,
                        default=1
                        )
    parser.add_argument('--epsilon', '-e',
                        type=float,
                        default=0.2
                        )
    parser.add_argument('--show_arg', '-s',
                        action=argparse.BooleanOptionalAction,
                        default=False
                        )
    parser.add_argument('--dir', '-d',
                        type=str,
                        default='data')
    args = parser.parse_args()
    config = get_config(args.config)(args.ue, args.bs, args.mu, args.epsilon)
    config.agent_name = args.agent_name

    global save_dir
    save_dir = os.path.join(os.getcwd(), args.dir)

    #  if args.show_arg:
    #      pprint.pprint(config.__dict__)
    #      pprint.pprint(config.compute.__dict__)
    #      pprint.pprint(config.channel.__dict__)

    return config


def main():
    set_seed(0)
    config = configuration()

    if config.agent_name == 'proposed':
        file_name = f'{config.agent_name}_{config.num_UE}_{config.num_BS}_{str(config.compute.mu).replace(".", "d")}.mat'
    else:
        file_name = f'{config.agent_name}_{config.num_UE}_{config.num_BS}_{str(config.compute.epsilon).replace(".", "d")}.mat'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    if os.path.exists(save_path):
        print(f'File {file_name} already exists. Return.')
        return
    data = {}
    io.savemat(save_path, data)  # Call dibs on the file.

    # Data
    data = {
        'latency': 0.0,
        'latency_commun': 0.0,
        'latency_compute_UE': 0.0,
        'latency_compute_BS': 0.0,
        'energy_consumption_local': 0.0,
        'energy_consumption_edge': 0.0,
        'energy_consumption': 0.0,
        'service_stats': 0.0,  # Computing location ratio.
        'energy_dead': np.zeros((config.num_step)),  # Device death time.
    }

    for n in tqdm(range(config.num_iter)):
        set_seed(n)
        if config.agent_name == 'random':
            agent = RandAgent(config)
        elif config.agent_name == 'maxsinr':
            agent = MaxSINRAgent(config)
        elif config.agent_name == 'maxcompute':
            agent = MaxComputeAgent(config)
        elif config.agent_name == 'combined':
            agent = CombinedSINRComputeAgent(config)
        elif config.agent_name == 'proposed':
            agent = ProposedAgent(config)
        else:
            raise ValueError(f'Invalid agent name: {config.agent_name}')
        env = CommunNet(agent, config)
        for _ in tqdm(range(config.num_step), leave=False):
            terminate = env.proceed()
            if terminate:
                # break
                pass
        data = process_data(data, env.data, n)
        print(f"\nAverage latency so far: {data['latency']}")
        env.summary()

    io.savemat(save_path, data)
    print(f'Data saved to: {save_path}')
    print(f'service_stats: {data["service_stats"]}')


if __name__ == '__main__':
    main()
