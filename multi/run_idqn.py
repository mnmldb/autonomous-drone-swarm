#--------- Training ---------#

# parameters
size = 10
fov = 3
n_agents = 3
train_episodes = 5000
max_steps = size * size

r = 0.0005
batch_size = 32
gamma = 0.99
buffer_limit = 50000
log_interval = 20
max_episodes = 30000
max_epsilon = 0.9
min_epsilon = 0.1
warm_up_steps = 2000
update_iter = 10
coverage_threshold = 0.95


# record metrics for each episode
time_steps = [] # number of time steps in total
epsilons = [] # epsilon at the end of each episode
greedy = [] # the ratio of greedy choices
coverage = [] # the ratio of visited cells at the end
speed = [] # number of time steps to cover decent amount of cells
results_mapping = [] # mapping status
results_agents = [] # mapping status with agent positions
results_count = [] # count status
total_reward = []


# initialize the environment
env = GridMultiAgent(x_size=size, y_size=size, fov_x=fov, fov_y=fov, n_agents=n_agents)

# independent DQN
memory = ReplayBuffer(buffer_limit)

q = QNet(env.observation_space, env.action_space)
q_target = QNet(env.observation_space, env.action_space)
q_target.load_state_dict(q.state_dict())
optimizer = optim.Adam(q.parameters(), lr=lr)

for episode_i in range(max_episodes):
    score = np.zeros(env.n_agents)
    episode_step = 0
    epsilon = max(min_epsilon, max_epsilon - (max_epsilon - min_epsilon) * (episode_i / (0.4 * max_episodes)))
    state = env.reset()
    done = [False for _ in range(env.n_agents)]
    while not all(done):
        action = q.sample_action(torch.Tensor(state).unsqueeze(0), epsilon)[0].data.cpu().numpy().tolist()
        next_state, reward, done, info = env.step(action)
        memory.put((state, action, (np.array(reward)).tolist(), next_state, np.array(done, dtype=int).tolist()))
        score += np.array(reward)
        state = next_state
        # max time step
        if episode_step == max_steps - 1:
            break
        episode_step += 1

    # training
    if memory.size() > warm_up_steps:
        train(q, q_target, memory, optimizer, gamma, batch_size, update_iter)

    # sync parameters
    if episode_i % log_interval == 0 and episode_i != 0:
        q_target.load_state_dict(q.state_dict())
    
    # record
    epsilons.append(epsilon)
    coverage.append(env.get_coverage())
    # greedy.append(greedy_count / (step + 1)) # multiply by step
    results_mapping.append(env.grid_status)
    results_agents.append(env.grid_agents_status)
    results_count.append(env.grid_counts)
    total_reward.append(score.sum())

    print('//Episode {0}//    Epsilon: {1:.3f},    Steps: {2},    Greedy Choices　(%): {3:.3f},    Coverage (%): {4:.3f},    Steps to Visit {5}% Cells: {6},    Total Reward: {7}'\
          .format(episode_i+1, epsilon, episode_step+1, 0, coverage[episode_i], coverage_threshold * 100, 0, total_reward[episode_i]))


#--------- Test ---------#

# parameters
epsilon = 0.1
test_episodes = 100
max_steps_test = size * size

# record metrics for each episode
time_steps_test = [] # number of time steps in total
coverage_test = [] # the ratio of visited cells at the end
speed_test = [] # number of time steps to cover decent amount of cells
results_mapping_test = [] # mapping status
results_agents_test = [] # mapping status with agent positions
results_count_test = [] # count status
total_reward_test = []

# initialize the environment
test_env = GridMultiAgent(x_size=size, y_size=size, fov_x=fov, fov_y=fov, n_agents=n_agents)

for episode_i in range(test_episodes):
    score = np.zeros(test_env.n_agents)
    episode_step = 0
    state = test_env.reset()
    done = [False for _ in range(test_env.n_agents)]
    while not all(done):
        action = q.sample_action(torch.Tensor(state).unsqueeze(0), epsilon)[0].data.cpu().numpy().tolist()
        next_state, reward, done, info = test_env.step(action)
        score += np.array(reward)
        state = next_state
        # max time step
        if episode_step == max_steps_test - 1:
            break
        episode_step += 1
    
    # record
    coverage_test.append(test_env.get_coverage())
    results_mapping_test.append(test_env.grid_status)
    results_agents_test.append(test_env.grid_agents_status)
    results_count_test.append(test_env.grid_counts)
    total_reward_test.append(score.sum())

    print('//Episode {0}//    Epsilon: {1:.3f},    Steps: {2},    Greedy Choices　(%): {3:.3f},    Coverage (%): {4:.3f},    Steps to Visit {5}% Cells: {6},    Total Reward: {7}'\
          .format(episode_i+1, epsilon, episode_step+1, 0, coverage_test[episode_i], coverage_threshold * 100, 0, total_reward_test[episode_i]))
