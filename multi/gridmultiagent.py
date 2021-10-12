class GridMultiAgent(gym.Env):
    metadata = {'render.modes': ['console']}
    # action id
    XM = 0 # x minus
    XP = 1 # x plus
    YM = 2 # y minus
    YP = 3 # y plus
    XMYM = 4 # x minus, y minus
    XMYP = 5 # x minus, y plus
    XPYM = 6 # x plus, y minus
    XPYP = 7 # x plus, y plus

    # grid status
    OOE = -2 # out of the environment
    OBS = -1 # obstacle
    POI = 0 # POI that is not mapped
    MAP = 1 # POI that is mapped
    AGT = 2 # agent
    
    def __init__(self, x_size=10, y_size=10, fov_x=3, fov_y=3, n_agents=3):
        super(GridMultiAgent, self).__init__()
        
        # size of 2D grid
        self.x_size = x_size
        self.y_size = y_size

        # number of the agents
        self.n_agents = n_agents
        
        # initialize mapping status
        self.init_grid()
        
        # initialize agent positions
        self.init_agent()
        
        # define action space
        n_actions = 8
        self.action_space = MultiAgentActionSpace([spaces.Discrete(n_actions) for _ in range(self.n_agents)])
        
        # define observation space
        self.fov_x = fov_x # number of cells around the agent
        self.fov_y = fov_y # number of cells around the agent
        self.obs_low = np.ones(self.fov_x * self.fov_y) * self.OOE
        self.obs_high = np.ones(self.fov_x * self.fov_y) * self.AGT 
        self.observation_space = MultiAgentObservationSpace([spaces.Box(self.obs_low, self.obs_high) for _ in range(self.n_agents)])
    
    def init_grid(self):
        # initialize the mapping status
        self.grid_status = np.zeros([self.x_size, self.y_size])
        self.grid_counts = np.zeros([self.x_size, self.y_size])

        ## randomly set obstacles
        n_obstacle = random.randrange(0, self.x_size * self.x_size * 0.2) # at most 20% of the grid
        for i in range(n_obstacle):
            x_obstacle = random.randrange(1, self.x_size - 1)
            y_obstacle = random.randrange(1, self.y_size - 1)
            self.grid_status[x_obstacle, y_obstacle] = self.OBS # -1
            self.grid_counts[x_obstacle, y_obstacle] = self.OBS # -1
        
        # number of POI in the environment (0)
        self.n_poi = self.x_size * self.y_size - np.count_nonzero(self.grid_status)

        # grid status and agent positions
        self.grid_agents_status = copy.deepcopy(self.grid_status)
    
    def init_agent(self):
        self.agent_pos = []
        # initialize the agent position
        for i in range(self.n_agents):
            while True:
                agent_pos_x = random.randrange(0, self.x_size)
                agent_pos_y = random.randrange(0, self.y_size)
                # avoid to initialize agent positions on obstacles and other agents potions
                if self.grid_agents_status[agent_pos_x, agent_pos_y] == self.POI: # 0
                    self.agent_pos.append([agent_pos_x, agent_pos_y])
                    self.grid_agents_status[agent_pos_x, agent_pos_y] = self.AGT # 2
                    break

    def grid_overlay(self):
        # copy grid status
        self.grid_agents_status = copy.deepcopy(self.grid_status)

        # overlay the latest agent positions
        for i in range(self.n_agents):
            agent_pos_x = self.agent_pos[i][0]
            agent_pos_y = self.agent_pos[i][1]
            self.grid_agents_status[agent_pos_x, agent_pos_y] = self.AGT # 2
    
    def get_coverage(self):
        mapped_poi = np.count_nonzero(self.grid_status == 1) + np.count_nonzero(self.grid_status == 2)
        return mapped_poi / self.n_poi

    def get_agent_obs(self):
        self.agent_obs = []
        # observation for each agent
        for agent in range(self.n_agents):
            # default: out of the gri
            single_obs = np.ones([self.fov_x, self.fov_y]) * self.OOE # -2
            for i in range(self.fov_x): # 0, 1, 2
                for j in range(self.fov_y): # 0, 1, 2
                    obs_x = self.agent_pos[agent][0] + (i - 1) # -1, 0, 1
                    obs_y = self.agent_pos[agent][1] + (j - 1) # -1, 0, 1
                    if obs_x >= 0 and obs_y >= 0 and obs_x <= self.x_size - 1 and obs_y <= self.y_size - 1:
                        single_obs[i][j] = self.grid_agents_status[obs_x][obs_y] # use grid_agents_status to capture other agent positions
            single_obs_flat = single_obs.flatten() # convert matrix to list
            self.agent_obs.append(single_obs_flat)
        return self.agent_obs

    def reset(self):
        # initialize the mapping status
        self.init_grid()
        # initialize the position of the agent
        self.init_agent()
        return self.get_agent_obs()
        
    def step(self, action):
        org_pos = copy.deepcopy(self.agent_pos) # original positions
        new_pos = copy.deepcopy(self.agent_pos) # new positions
        reward = []

        # move all agents to new positions
        for i in range(self.n_agents):
            # action
            single_action = action[i]

            # move the agent
            if single_action == self.XM: # 0
                new_pos[i][0] -= 1
            elif single_action == self.XP: # 1
                new_pos[i][0] += 1
            elif single_action == self.YM: # 2
                new_pos[i][1] -= 1
            elif single_action == self.YP: # 3
                new_pos[i][1] += 1
            elif single_action == self.XMYM: # 4
                new_pos[i][0] -= 1
                new_pos[i][1] -= 1
            elif single_action == self.XMYP: # 5
                new_pos[i][0] -= 1
                new_pos[i][1] += 1
            elif single_action == self.XPYM: # 6
                new_pos[i][0] += 1
                new_pos[i][1] -= 1
            elif single_action == self.XPYP: # 7
                new_pos[i][0] += 1
                new_pos[i][1] += 1
            else:
                raise ValueError("Received invalid action={} which is not part of the action space".format(single_action))
        
            # account for the boundaris of the grid (-2: out of the grid)
            if new_pos[i][0] > self.x_size - 1 or new_pos[i][0] < 0 or new_pos[i][1] > self.y_size - 1 or new_pos[i][1] < 0:
                # go back to the original position
                new_pos[i][0] = org_pos[i][0]
                new_pos[i][1] = org_pos[i][1]
                # self.grid_counts[new_pos[i][0], new_pos[i][1]] += 1
                single_reward = -10
            # previous status of the cell
            else:
                prev_status = self.grid_status[new_pos[i][0], new_pos[i][1]]
                if prev_status == self.OBS: # the new position is on the obstacle
                    # go back to the original position
                    new_pos[i][0] = org_pos[i][0]
                    new_pos[i][1] = org_pos[i][1]
                    # self.grid_counts[new_pos[i][0], new_pos[i][1]] += 1
                    single_reward = -10
                elif prev_status == self.POI:
                    # update the cell 
                    self.grid_status[new_pos[i][0], new_pos[i][1]] = 1
                    # self.grid_counts[new_pos[i][0], new_pos[i][1]] += 1
                    single_reward = 10
                elif prev_status == self.MAP:
                    # self.grid_counts[new_pos[i][0], new_pos[i][1]] += 1
                    single_reward = -1
            # reward of all agents    
            reward.append(single_reward)
        
        # first collision check
        intermediate_pos = copy.deepcopy(new_pos)
        first_revert = []
        not_revert = []
        ## unique new positions
        seen = []
        unique_new_pos = [x for x in new_pos if x not in seen and not seen.append(x)]
        ## go back to original positions when collision occurs
        if len(new_pos) != len(unique_new_pos):
            first_revert = []
            for i in range(self.n_agents):
                cnt = 0
                for j in new_pos:
                    if new_pos[i] == j:
                        cnt += 1
                if cnt > 1:
                    intermediate_pos[i] = org_pos[i]
                    first_revert.append(i)
            not_revert = list(set(range(self.n_agents)) ^ set(first_revert))
        ## penalty
        for i in first_revert:
            reward[i] = -30
        
        # second collision check
        adjusted_pos = copy.deepcopy(intermediate_pos)
        second_revert = []
        ## unique intermediate positions
        seen = []
        unique_intermediate_pos = [x for x in intermediate_pos if x not in seen and not seen.append(x)]
        ## go back to original positions when collision occurs
        if len(intermediate_pos) != len(unique_intermediate_pos):
            for i in not_revert:
                cnt = 0
                for j in intermediate_pos:
                    if intermediate_pos[i] == j:
                        cnt += 1
                if cnt > 1:
                    adjusted_pos[i] = org_pos[i]
                    second_revert.append(i)
        ## penalty
        for i in second_revert:
            reward[i] = -30

        # are we map all cells?
        mapped_poi = np.count_nonzero(self.grid_status == 1)
        single_done = bool(mapped_poi == self.n_poi)
        done = [single_done] * self.n_agents

        # optionally we can pass additional info
        info = {}

        # update agent positions
        self.agent_pos = adjusted_pos

        # update the grid_agents_status
        self.grid_overlay()
        
        return self.get_agent_obs(), reward, done, info

    def close(self):
        pass