### CUSP-GX-5006: Urban Science Intensive II (Fall 2021) 
# Developing Autonomous Drone Swarms with Multi-Agent Reinforcement Learning for Scalable Post-Disaster Damage Assessment
<table align="center">
<tr>
<th>Efficient Case</th>
<th>Reduncant Case</th>
<th>Unsuccessful Case</th>
</tr>
<tr>
<td><img src="https://user-images.githubusercontent.com/47055092/142976224-7ed678fe-4270-4576-ab9a-fb7f0054729e.gif" width="240px"></td>
<td><img src="https://user-images.githubusercontent.com/47055092/142976291-ccb563f6-cf50-456a-ba41-ee71dfb68999.gif" width="240px"></td>
<td><img src="https://user-images.githubusercontent.com/47055092/142979550-21985f8f-a8f6-46c1-9941-604d4953be32.gif" width="240px"></td>
</tr>
</table>

## Abstract
In recent years, drones have been used for supporting post-disaster damage assessment of buildings, roads, and infrastructure. Given that human resources are limited post-disaster, the development of autonomous drone swarms rather than a single drone can enable a further rapid and thorough assessment. Multi-Agent Reinforcement Learning is a promising approach because it can adapt to dynamic environments and relax computational complexity for optimization. This research project applies Multi-Agent Q-learning to autonomous navigation for mapping and multi-objective drone swarm exploration of a disaster area in terms of tradeoffs between coverage and scalability. We compare two different state spaces by assessing coverage of the environment within the given limited time. We find that using drones’ observations of the mapping status within their Field of View as a state is better than using their positions in terms of coverage and scalability. Also, we empirically specify parametric thresholds for performance of a multi-objective RL algorithm as a point of reference for future work incorporating deep learning approaches, e.g., Graph Neural Networks (GNNs).

## Team Members
Daisuke Nakanishi, Gurpreet Singh, Kshitij Chandna 

## Folder Structure
~~~
.
├── experiment   # Jupyter notebooks to run the algorithms
├── vis          # Jupyter notebooks to create animations
├── single       # RL environment for single agent
├── multi        # RL environment for multi agents
├── QL           # Tabular Q-learning
├── QL_NN        # (Preliminary) Function approximation Q-learning
├── DQN          # (Preliminary) Deep Q-Networks
├── GNN          # (Preliminary) Graph Neural Networks
├── .gitignore
└── README.md
~~~

## Methods
### Environment
<img src="https://user-images.githubusercontent.com/47055092/143366238-53ff4fa1-7de5-4837-a874-4348e10b0389.png" width="240px">

We discretize the 2-dimensional mission environment (disaster area) into a grid consisting of m × m square cells. The length of the square cell side is sufficiently larger than the size of the drone, and two or more drones can occupy a single cell . The cell visited by at least one drone is considered mapped.

### Sequential Decision-Making
<img src="https://user-images.githubusercontent.com/47055092/143366712-f95833fe-5708-4d22-91ba-8f7c1249a802.png" height="180px" width="auto">

Each time step, n drones take action sequentially; the preceding drone mapping results are reflected in the mapping status matrix M before the following one decides its action. Therefore, the following drone could indirectly and partially observe the preceding ones’ behaviors. The order of drones taking action is randomly set in each time step.

### State Space and Action Space
<img src="https://user-images.githubusercontent.com/47055092/143368985-16eaff34-dc1d-4831-8114-e6868540b2b5.png" height="140px" width="auto">

We examine two state spaces to deal with possible disaster areas and scenarios. Action space consists of four possible directions, A={up,down,right,left}.

## Results
Our Webpage: https://1312gurpreet.github.io/droneswarm/index.html

## Dependencies
- [OpenAI Gym](https://github.com/openai/gym)==0.21.0 or newer
- Python==3.7.12 or newer
- (Optional) [Stable Baselines3](https://stable-baselines.readthedocs.io/en/master/index.html#)==1.3.0 or newer