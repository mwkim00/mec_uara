# Distributed Task Offloading in MEC Networks

Source code for the paper ["Distributed Task Offloading and Resource Allocation for Latency Minimization in Mobile Edge Computing Networks"](https://ieeexplore.ieee.org/document/10675431).
## Usage

```
python main.py \
    -c CONFIG_NAME \
    -a AGENT_NAME \
    -u NUM_UE \
    -b NUM_BS \
    -m MU \
    -e EPSILON \
    -s SHOWARGS \
    -d DIR
```

### Parameters
- `CONFIG_NAME`: Configuration name. (default: `config`)
- `AGENT_NAME`: Agent name. (default: `proposed`)
- `NUM_UE`: Number of UEs. (default: `80`)
- `NUM_BS`: Number of BSs. (default: `4`)
- `MU`: Learning rate. (default: `1`)
- `EPSILON`: Epsilon for epsilon-greedy. (default: `0.2`)
- `SHOWARGS`: Show arguments. (default: `False`)
- `DIR`: Directory for saving results. (default: `data`)

## Notation differences

|   Code  | Paper |
|:-------:|:-----:|
|    mu   | alpha |
| nu[0,:] |  mu   |
| nu[1,:] |  nu   |

## Citation

If you find this code useful, please consider citing our paper:

```
@ARTICLE{10675431,
  author={Kim, Minwoo and Jang, Jonggyu and Choi, Youngchol and Yang, Hyun Jong},
  journal={IEEE Transactions on Mobile Computing}, 
  title={Distributed Task Offloading and Resource Allocation for Latency Minimization in Mobile Edge Computing Networks}, 
  year={2024},
  volume={23},
  number={12},
  pages={15149-15166},
  keywords={Optimization;Servers;Artificial intelligence;Resource management;Batteries;Energy consumption;Delays;Latency minimization;delay minimization;mobile edge computing;resource allocation;user association;task offloading;energy efficiency;edge AI},
  doi={10.1109/TMC.2024.3458185}}
```


