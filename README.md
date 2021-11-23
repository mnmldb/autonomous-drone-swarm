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
In recent years, drones have been used for supporting post-disaster damage assessment of buildings, roads, and infrastructure. Given that human resources are limited post-disaster, the development of autonomous drone swarms rather than a single drone is necessary for a further rapid and thorough assessment. Multi-Agent Reinforcement Learning is a promising approach because it can work without prior knowledge of a disaster area. In this research, we use Multi-Agent Q-learning to develop an autonomous navigation algorithm in a mapping task; drones should visit points of interest within a disaster area as fast and thoroughly as possible. We compare different state spaces by assessing coverage of the environment within the given limited time. We observe that using drones’ observations of the mapping status within their Field of View as a state is a better solution in terms of coverage and scalability. The performance of our algorithm can serve as a baseline for future work that would reduce redundant drones’ actions even further, incorporating deep learning approaches such as Graph Neural Network.

