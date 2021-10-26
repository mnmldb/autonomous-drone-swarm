# ===================================================================================================
# Training
# ===================================================================================================

# records for each episode
time_steps = [] # number of time steps in total
epsilons = [] # epsilon at the end of each episode
greedy = [] # the ratio of greedy choices
coverage = [] # the ratio of visited cells at the end
speed = [] # number of time steps to cover decent amount of cells
sum_q_values = [] # sum of q-values
results_mapping = [] # mapping status
results_count = [] # count status
total_reward = []
total_action_values = []
total_greedy_action_values = []

q_class = []

coverage_threshold = 0.90
max_stuck = 3

# parameters for training
train_episodes = 3000
max_steps = 10 * 10 * 3

# initialize the environment and the q tables
env = Grid(x_size=10, y_size=10, n_agents=2, fov_x=3, fov_y=3)
q = QTables(observation_space=env.observation_space, action_space=env.action_space, eps_start=1, eps_end=0.1, gamma=0.5, r=0.999, lr=0.1)

# training
for episode in range(train_episodes):
    state = env.reset()
    state = [arr.astype('int') for arr in state] # convert from float to integer

    greedy_count = [0] * env.n_agents
    coverage_track = True
    epi_reward = [0] * env.n_agents
    epi_action_value = [0] * env.n_agents
    epi_greedy_action_value = [0] * env.n_agents

    for step in range(max_steps):
        action_order = random.sample(env.idx_agents, env.n_agents) # return a random order of the drone indices
        for agent_i in action_order:
            action, greedy_tf, action_value = q.get_action(observations=state, agent_i=agent_i, stuck_counts=env.stuck_counts, max_stuck=max_stuck, e_greedy=True)
            next_state, reward, done = env.step(action, agent_i)
            next_state = [arr.astype('int') for arr in next_state] # convert from float to integer
            q.train(state, next_state, action, reward, done, agent_i)

            epi_reward[agent_i] += reward
            greedy_count[agent_i] += greedy_tf * 1
            epi_action_value[agent_i] += action_value
            epi_greedy_action_value[agent_i] += action_value * greedy_tf

            if done:
                break
        
            # update the observation
            state = next_state

        # check if decent amoung of cells are visited
        current_coverage = env.get_coverage()
        if current_coverage >= coverage_threshold and coverage_track:
            speed.append(step)
            coverage_track = False

        # check if the task is completed
        if done:
            time_steps.append(step)
            break
        elif step == max_steps - 1:
            time_steps.append(step)
            if coverage_track:
                speed.append(np.nan)

    # record
    time_steps.append(step + 1)
    epsilons.append(q.eps)
    coverage.append(env.get_coverage())
    greedy.append(list(map(lambda x: x / (step + 1), greedy_count)))
    sum_q_values.append([q.q_tables[0].sum(), q.q_tables[1].sum()])
    results_mapping.append(env.grid_status)
    results_count.append(env.grid_counts)
    total_reward.append(epi_reward)
    total_action_values.append(epi_action_value)
    total_greedy_action_values.append(epi_greedy_action_value)
    q_class.append(copy.deepcopy(q))

    # update epsilon
    q.update_eps()

    print('//Episode {0}//    Epsilon: {1:.3f},    Steps: {2},    Greedy Choices　(%): {3:.3f},    Coverage (%): {4:.3f},    Steps to Visit {5}% Cells: {6},    Sum of Q-Values: {7:.1f}, {8:.1f}    Total Reward: {9}'\
          .format(episode+1, q.eps, step+1, np.mean(greedy[episode]), coverage[episode], coverage_threshold * 100, speed[episode], sum_q_values[episode][0], sum_q_values[episode][1], np.mean(total_reward[episode])))


# ===================================================================================================
# Test
# ===================================================================================================

# fixed obstacle positions and initial drone positions
test_grid = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  -1.,  -1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  -1.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  -1.,  -1.,  -1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

test_agent_pos = [[0, 0], [9, 9]]

# records for each episode
time_steps_test = [] # number of time steps in total
epsilons_test = [] # epsilon at the end of each episode
greedy_test = [] # the ratio of greedy choices
coverage_test = [] # the ratio of visited cells at the end
speed_test = [] # number of time steps to cover decent amount of cells
sum_q_values_test = [] # sum of q-values
results_mapping_test = [] # mapping status
results_count_test = [] # count status
total_reward_test = []
total_action_values_test = []
total_greedy_action_values_test = []

coverage_threshold_test = 0.90
max_stuck_test = 10000 # never random actions

# parameters for training
test_episodes = 3000
max_steps_test = 10 * 10 * 3

# initialize the environment and the q table
env_test = Grid(x_size=10, y_size=10, n_agents=2, fov_x=3, fov_y=3)
env_test.grid_status = test_grid # fixed obstacle positions
env_test.agent_pos = test_agent_pos # fixed initial drone positions

# training
for episode in range(test_episodes):
    _ = env.reset()
    env_test.grid_status = copy.deepcopy(test_grid)
    env_test.agent_pos = copy.deepcopy(test_agent_pos)
    state = env_test.get_agent_obs()
    state = [arr.astype('int') for arr in state] # convert from float to integer

    greedy_count_test = [0] * env_test.n_agents
    coverage_track_test = True
    epi_reward_test = [0] * env_test.n_agents
    epi_action_value_test = [0] * env_test.n_agents
    epi_greedy_action_value_test = [0] * env_test.n_agents

    for step in range(max_steps_test):
        for agent_i in range(env_test.n_agents): # fixed order of actions
            action, greedy_tf, action_value = q_class[episode].get_action(observations=state, agent_i=agent_i, stuck_counts=env_test.stuck_counts, max_stuck=max_stuck_test, e_greedy=False)
            next_state, reward, done = env_test.step(action, agent_i)
            next_state = [arr.astype('int') for arr in next_state] # convert from float to integer

            epi_reward_test[agent_i] += reward
            greedy_count_test[agent_i] += greedy_tf
            epi_action_value_test[agent_i] += action_value
            epi_greedy_action_value_test[agent_i] += action_value * greedy_tf

            if done:
                break
        
            # update the observation
            state = next_state

        # check if decent amoung of cells are visited
        current_coverage_test = env_test.get_coverage()
        if current_coverage_test >= coverage_threshold_test and coverage_track_test:
            speed_test.append(step)
            coverage_track_test = False

        # check if the task is completed
        if done:
            time_steps_test.append(step)
            break
        elif step == max_steps_test - 1:
            time_steps_test.append(step)
            if coverage_track_test:
                speed_test.append(np.nan)

    # record
    time_steps_test.append(step + 1)
    epsilons_test.append(q.eps)
    coverage_test.append(env_test.get_coverage())
    greedy_test.append(list(map(lambda x: x / (step + 1), greedy_count_test)))
    sum_q_values_test.append([q_class[episode].q_tables[0].sum(), q_class[episode].q_tables[1].sum()])
    results_mapping_test.append(env_test.grid_status)
    results_count_test.append(env_test.grid_counts)
    total_reward_test.append(epi_reward_test)
    total_action_values_test.append(epi_action_value_test)
    total_greedy_action_values_test.append(epi_greedy_action_value_test)

    print('//Episode {0}//    Epsilon: {1:.3f},    Steps: {2},    Greedy Choices　(%): {3:.3f},    Coverage (%): {4:.3f},    Steps to Visit {5}% Cells: {6},    Sum of Q-Values: {7:.1f}, {8:.1f}    Total Reward: {9}'\
          .format(episode+1, 0, step+1, np.mean(greedy_test[episode]), coverage_test[episode], coverage_threshold_test * 100, speed_test[episode], sum_q_values_test[episode][0], sum_q_values_test[episode][1], np.mean(total_reward_test[episode])))
