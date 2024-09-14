"""
Run main.py for different scenarios.
To avoid overwriting result files, the data is saved in a folder named by the current time.
For test configuration, run main.py instead.
    python main.py  --config test --agent_name AGENT_NAME --ue UE --bs BS --mu MU --dir DIR
"""
import os
import subprocess
from datetime import datetime


start_time = datetime.now()

# Default values. Add lines underneath to overwrite and test different scenarios.
show_arg = False
ues = [20 * i for i in range(1, 9)]
bss = [4, 8, 12]
configs = ['base', 'highcommun', 'highcompute']
agents = ['random', 'maxsinr', 'maxcompute', 'combined', 'proposed']
mus = [0, 1, 20, 40, 60, 80, 100]
# Offloading ratio for comparison schemes
epsilons = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for config in configs:
    # save_dir = os.path.join(os.getcwd(), 'finetune', datetime.now().strftime("%Y%m%d%H%M%S"), config)
    save_dir = os.path.join(os.getcwd(), 'data', config)  # 'data', 'highcommun', 'highcompute'
    os.makedirs(save_dir, exist_ok=True)

    for bs in bss:
        for ue in ues:
            for agent in agents:
                if agent == 'proposed':
                    for mu in mus:
                        print(f"\n=====================================================\n"
                              f"Running {agent} for {ue} UEs and {bs} BSs with mu={mu}\n"
                              f"=====================================================\n")
                        #  show = '--no-show_arg' if not show_arg else '--show_arg'
                        p = subprocess.Popen(["python", os.path.join(os.getcwd(), 'main.py'), '--config', config,
                                              '--agent_name', str(agent), '--ue', str(ue), '--bs', str(bs), '--mu',
                                              str(mu), '--dir', str(save_dir)])
                        # Wait for the subprocess to end. Without this, the program ends before the subprocess is finished.
                        exit_codes = p.wait()
                else:
                    for epsilon in epsilons:
                        print(f"\n=====================================================\n"
                              f"Running {agent} for {ue} UEs and {bs} BSs with epsilon={epsilon}\n"
                              f"=====================================================\n")
                        #  show = '--no-show_arg' if not show_arg else '--show_arg'
                        p = subprocess.Popen(["python", os.path.join(os.getcwd(), 'main.py'), '--config', config,
                                              '--agent_name', str(agent), '--ue', str(ue), '--bs', str(bs), '--epsilon', str(epsilon),
                                              '--dir', str(save_dir)])
                        # Wait for the subprocess to end. Without this, the program ends before the subprocess is finished.
                        exit_codes = p.wait()

end_time = datetime.now()

print(f"\n=====================================================\n"
      f"Completed. Elapsed time: {end_time.replace(microsecond=0) - start_time.replace(microsecond=0)}\n"
      f"=====================================================\n")
