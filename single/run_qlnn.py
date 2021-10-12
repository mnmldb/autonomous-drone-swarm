#--------- Training ---------#

# metrics for each episode
time_steps = [] # number of time steps in total
epsilons = [] # epsilon at the end of each episode
greedy = [] # the ratio of greedy choices
coverage = [] # the ratio of visited cells at the end
speed = [] # number of time steps to cover decent amount of cells
sum_q_values = [] # sum of q-values
results_mapping = [] # mapping status
results_count = [] # count status
total_reward = []
coverage_threshold = 0.95 #

# parameters for training
train_episodes = 5000
max_steps = 200

# initialize the environment
env_train = Grid(x_size=5, y_size=5, fov_x=3, fov_y=3)

# initialize the neural network
n_obs = env_train.observation_space.shape[0]
n_mid = 32
n_action = env_train.action_space.n
net = Net(n_obs, n_mid, n_action)
loss_fnc = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
is_gpu = False
model = QValues(n_obs, n_action, net, loss_fnc, optimizer , is_gpu, eps=1, gamma=0.9, r=0.999, lr=0.01)

# training
for episode in range(train_episodes):
    obs = env_train.reset()
    # state = [arr.astype('int') for arr in state] # convert from float to integer

    greedy_count = 0 # shared with all agents
    coverage_track = True
    epi_reward = 0

    for step in range(max_steps):
        action, greedy_tf = model.get_action(obs)
        next_obs, reward, done, info = env_train.step(action)
        # next_state = [arr.astype('int') for arr in next_state] # convert from float to integer
        model.train(obs, next_obs, action, reward, done)
        greedy_count += sum([greedy_tf])
        epi_reward += reward

        # check if decent amoung of cells are visited
        current_coverage = env_train.get_coverage()
        if current_coverage >= coverage_threshold and coverage_track:
            speed.append(step)
            coverage_track = False

        # check if the task is completed
        if done:
            time_steps.append(step)
            model.update_eps()
            break
        elif step == max_steps - 1:
            time_steps.append(step)
            model.update_eps()
            if coverage_track:
                speed.append(np.nan)
        
        # update the observation
        obs = next_obs
    
    epsilons.append(model.eps)
    coverage.append(env_train.get_coverage())
    greedy.append(greedy_count / (step + 1)) # multiply by step
    results_mapping.append(env_train.grid_status)
    results_count.append(env_train.grid_counts)
    total_reward.append(epi_reward)


    print('//Episode {0}//    Epsilon: {1:.3f},    Steps: {2},    Greedy Choices　(%): {3:.3f},    Coverage (%): {4:.3f},    Steps to Visit {5}% Cells: {6},    Total Reward {7}'\
          .format(episode+1, model.eps, step+1, greedy[episode], coverage[episode], coverage_threshold * 100, speed[episode], total_reward[episode]))



#--------- Test ---------#

# metrics for each episode
time_steps_test = [] # number of time steps in total
coverage_test = [] # the ratio of visited cells at the end
speed_test = [] # number of time steps to cover decent amount of cells
results_mapping_test = [] # mapping status
results_count_test = [] # count status
total_reward = []
coverage_threshold = 0.95

# parameters for training
train_episodes = 100
max_steps = 5 * 5 * 2

# initialize the environment
env_test = Grid(x_size=5, y_size=5, fov_x=3, fov_y=3)

for episode in range(train_episodes):
    obs = env_test.reset()

    greedy_count = 0 # shared with all agents
    coverage_track = True
    epi_reward = 0

    for step in range(max_steps):
        action, greedy_tf = model.get_action(obs)
        next_obs, reward, done, info = env_test.step(action)
        greedy_count += sum([greedy_tf])
        epi_reward += reward

        # check if decent amoung of cells are visited
        current_coverage = env_test.get_coverage()
        if current_coverage >= coverage_threshold and coverage_track:
            speed_test.append(step)
            coverage_track = False

        # check if the task is completed
        if done:
            time_steps_test.append(step)
            break
        elif step == max_steps - 1:
            time_steps_test.append(step)
            if coverage_track:
                speed_test.append(np.nan)
        
        # update the observation
        obs = next_obs
    
    coverage_test.append(env_test.get_coverage())
    results_mapping_test.append(env_test.grid_status)
    results_count_test.append(env_test.grid_counts)
    total_reward.append(epi_reward)

    print('//Episode {0}//    Steps: {1},    Coverage (%): {2:.3f},    Steps to Visit {3}% Cells: {4},    Total Reward: {5}'\
        .format(episode+1,  step+1,  coverage_test[episode], coverage_threshold * 100, speed_test[episode], total_reward[episode]))

print(np.mean(coverage_test))