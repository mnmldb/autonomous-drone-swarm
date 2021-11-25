class QTables():
    def __init__(self, observation_space, action_space, eps_start=1, eps_end=0.1, gamma=0.9, r=0.99, lr=0.1):
        self.num_agents = len(observation_space)

        self.observation_space = observation_space
        self.observation_length = observation_space[0].shape[0]
        self.size = int(self.observation_space[0].high[0] - self.observation_space[0].low[0]) + 1

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
            self.q_tables.append(np.zeros([self.size**2, self.action_num]))

        self.q_tables_count = []
        for agent_i in range(self.num_agents):
            self.q_tables_count.append(np.zeros([self.size**2, self.action_num]))

    # support function: convert the fov to the unique row number in the q table
    def obs_to_row(self, obs_array):
        return obs_array[0] * self.size + obs_array[1]
    
    def get_action(self, obs, i):
        if np.random.rand() < self.eps:
            action = random.choice(self.action_values)
            greedy = False
        else:
            obs_row = self.obs_to_row(obs[i])
            action = np.argmax(self.q_tables[i][obs_row])
            greedy = True
        
        return action, greedy
    
    def update_eps(self):
        # update the epsilon
        if self.eps > self.eps_end: # lower bound
            self.eps *= self.r

    def train(self, obs, obs_next, action, reward, done, i):
        obs_row = self.obs_to_row(obs[i])
        obs_next_row = self.obs_to_row(obs_next[i])

        q_current = self.q_tables[i][obs_row][action] # current q value
        q_next_max = np.max(self.q_tables[i][obs_next_row]) # the maximum q value in the next state

        # update the q value
        if done:
            self.q_tables[i][obs_row][action] = q_current + self.lr * reward
        else:
            self.q_tables[i][obs_row][action] = q_current + self.lr * (reward + self.gamma * q_next_max - q_current)
        
        # update the count
        self.q_tables_count[i][obs_row][action] += 1