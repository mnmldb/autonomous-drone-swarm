# ===================================================================================================
# Training: 1 drone
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
max_stuck = 100000

# parameters for training
train_episodes = 200000
max_steps = 10 * 10 * 2

# initialize the environment and the q tables
env = Grid(x_size=10, y_size=10, n_agents=1, fov_x=3, fov_y=3)
q = QTables(observation_space=env.observation_space, action_space=env.action_space, eps_start=1, eps_end=0, gamma=0.5, r=0.9999, lr=0.01)

# training
for episode in range(train_episodes):
    state = env.reset([[0, 0]])
    state = [arr.astype('int') for arr in state] # convert from float to integer
    eps_tmp = q.eps

    greedy_count = [0] * env.n_agents
    coverage_track = True
    epi_reward = [0] * env.n_agents
    epi_action_value = [0] * env.n_agents
    epi_greedy_action_value = [0] * env.n_agents

    for step in range(max_steps):
        action_order = random.sample(env.idx_agents, env.n_agents) # return a random order of the drone indices
        for agent_i in action_order:
            action, greedy_tf, action_value = q.get_action(observations=state, agent_i=agent_i, stuck_counts=env.stuck_counts, max_stuck=max_stuck, e_greedy=True, softmax=False)
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
    epsilons.append(eps_tmp)
    coverage.append(env.get_coverage())
    greedy.append(list(map(lambda x: x / (step + 1), greedy_count)))
    sum_q_values.append([q.q_tables[0].sum()])
    results_mapping.append(env.grid_status)
    results_count.append(env.grid_counts)
    total_reward.append(epi_reward)
    total_action_values.append(epi_action_value)
    total_greedy_action_values.append(epi_greedy_action_value)

    if episode % 1000 == 0:
        q_class.append(copy.deepcopy(q))

    # update epsilon
    q.update_eps()

    print('//Episode {0}//    Epsilon: {1:.3f},    Steps: {2},    Greedy Choices　(%): {3:.3f},    Coverage (%): {4:.3f},    Steps to Visit {5}% Cells: {6},    Sum of Q-Values: {7:.1f},    Total Reward: {8}'\
          .format(episode+1, eps_tmp, step+1, np.mean(greedy[episode]), coverage[episode], coverage_threshold * 100, speed[episode], sum_q_values[episode][0], np.mean(total_reward[episode])))