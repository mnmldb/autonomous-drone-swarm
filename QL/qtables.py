# multi-agent setting
# each agent has an individual q table

class QTables():
    def __init__(self, observation_space, action_space, eps_start=1, eps_end=0.1, gamma=0.9, r=0.99, lr=0.1):
        self.num_agents = len(observation_space)

        self.observation_space = observation_space
        self.observation_values = [-2, -1, 0, 1]
        self.observation_num = len(self.observation_values) # 4
        self.observation_length = observation_space[0].shape[0] # field of view

        self.action_space = action_space
        self.action_values = [0, 1, 2, 3] # corresponding to the column numbers in q table
        self.action_num = len(self.action_values) # 4

        self.eps = eps_start  # current epsilon
        self.eps_end = eps_end # epsilon lower bound
        self.r = r  # decrement rate of epsilon
        self.gamma = gamma  # discount rate
        self.lr = lr  # learning rate

        self.q_tables = []
        for agent_i in range(self.num_agents):
            self.q_tables.append(np.random.rand(self.observation_num**self.observation_length, self.action_num)) # (256 x 4) x num_agents
        
        self.q_table_counts = []
        for agent_i in range(self.num_agents):
            self.q_table_counts.append(np.zeros([self.observation_num**self.observation_length, self.action_num])) # (256 x 4) x num_agents

    # support function: convert the fov to the unique row number in the q table
    def obs_to_row(self, obs_array):
        obs_shift = map(lambda x: x + 2, obs_array) # add 1 to each element
        obs_power = [v * (self.observation_num ** i) for i, v in enumerate(obs_shift)] # apply exponentiation to each element
        return sum(obs_power) # return the sum (results are between 0 and 256)
    
    def get_action(self, observations, agent_i, stuck_counts, max_stuck, e_greedy=True):
        # convert the observation to a row number
        obs_row = self.obs_to_row(observations[agent_i])
        if stuck_counts[agent_i] >= max_stuck: # random action to avoid stuck
            action = random.choice(self.action_values)
            greedy = False
            action_value = self.q_tables[agent_i][obs_row][action]
        elif e_greedy: # epsilon greedy for training
            if np.random.rand() < self.eps:
                action = random.choice(self.action_values)
                greedy = False
                action_value = self.q_tables[agent_i][obs_row][action]
            else:
                action = np.argmax(self.q_tables[agent_i][obs_row])
                greedy = True
                action_value = self.q_tables[agent_i][obs_row][action]
        else: # all greedy choices for testing performance
            action = np.argmax(self.q_tables[agent_i][obs_row])
            greedy = True
            action_value = self.q_tables[agent_i][obs_row][action]
        
        return action, greedy, action_value
    
    def update_eps(self):
        # update the epsilon
        if self.eps > self.eps_end: # lower bound
            self.eps *= self.r

    def train(self, obs, obs_next, action, reward, done, agent_i):
        obs_row = self.obs_to_row(obs[agent_i])
        obs_next_row = self.obs_to_row(obs_next[agent_i])
        act_col = action

        q_current = self.q_tables[agent_i][obs_row][act_col] # current q value
        q_next_max = np.max(self.q_tables[agent_i][obs_next_row]) # the maximum q value in the next state

        # update the q value
        if done:
            self.q_tables[agent_i][obs_row][act_col] = q_current + self.lr * reward
        else:
            self.q_tables[agent_i][obs_row][act_col] = q_current + self.lr * (reward + self.gamma * q_next_max - q_current)

        # inclement the corresponding count
        self.q_table_counts[agent_i][obs_row][act_col] += 1