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
        self.idx_agents = list(range(n_agents)) # [0, 1, 2, ..., n_agents - 1]
        
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

        ## randomly set obstacles
        n_obstacle = random.randrange(0, self.x_size * self.x_size * 0.2) # at most 20% of the grid
        for i in range(n_obstacle):
            x_obstacle = random.randrange(1, self.x_size - 1)
            y_obstacle = random.randrange(1, self.y_size - 1)
            self.grid_status[x_obstacle, y_obstacle] = self.OBS # -1
        
        # initialize the count status
        self.grid_counts = np.tile(self.grid_status, (self.n_agents, 1)).reshape(self.n_agents, self.x_size, self.y_size)
        
        # number of POI in the environment (0)
        self.n_poi = self.x_size * self.y_size - np.count_nonzero(self.grid_status)

        # grid status and agent positions
        self.grid_agents_status = copy.deepcopy(self.grid_status)
    
    def init_agent(self):
        self.agent_pos = []
        self.stuck_counts = [0] * self.n_agents

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
        mapped_poi = np.count_nonzero(self.grid_status == 1)
        return mapped_poi / self.n_poi

    def get_agent_obs(self):
        self.agent_obs = []

        # observation for each agent
        for agent in range(self.n_agents):

            # default: out of the grid
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

        # check if the drones at initial positions are surrounded by obstacles
        while True:
            obs = self.get_agent_obs()
            obs_tf = []
            for i in range(self.n_agents):
                agent_obs_tf = obs[i][0] != 0 and obs[i][1] != 0 and obs[i][2] != 0 and obs[i][3] != 0
                obs_tf.append(agent_obs_tf)
            if any(obs_tf):
                self.init_grid()
                self.init_agent()
            else:
                break

        return self.get_agent_obs()
        
    def step(self, action, i):
        # original positions
        org_x = self.agent_pos[i][0]
        org_y = self.agent_pos[i][1]

        # move the agent
        if action == self.XM: # 0
            self.agent_pos[i][0] -= 1
        elif action == self.XP: # 1
            self.agent_pos[i][0] += 1
        elif action == self.YM: # 2
            self.agent_pos[i][1] -= 1
        elif action == self.YP: # 3
            self.agent_pos[i][1] += 1
        elif action == self.XMYM: # 4
            self.agent_pos[i][0] -= 1
            self.agent_pos[i][1] -= 1
        elif action == self.XMYP: # 5
            self.agent_pos[i][0] -= 1
            self.agent_pos[i][1] += 1
        elif action == self.XPYM: # 6
            self.agent_pos[i][0] += 1
            self.agent_pos[i][1] -= 1
        elif action == self.XPYP: # 7
            self.agent_pos[i][0] += 1
            self.agent_pos[i][1] += 1
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))
    
        # account for the boundaris of the grid (-2: out of the grid)
        if self.agent_pos[i][0] > self.x_size - 1 or self.agent_pos[i][0] < 0 or self.agent_pos[i][1] > self.y_size - 1 or self.agent_pos[i][1] < 0:
            # go back to the original position
            self.agent_pos[i][0] = org_x
            self.agent_pos[i][1] = org_y
            self.grid_counts[i][self.agent_pos[i][0], self.agent_pos[i][1]] += 1
            reward = 0

        # previous status of the cell
        else:
            prev_status = self.grid_status[self.agent_pos[i][0], self.agent_pos[i][1]]
            if prev_status == self.OBS: # the new position is on the obstacle
                # go back to the original position
                self.agent_pos[i][0] = org_x
                self.agent_pos[i][1] = org_y
                self.grid_counts[i][self.agent_pos[i][0], self.agent_pos[i][1]] += 1
                reward = 0
            if prev_status == self.AGT:
                # go back to the original position
                self.agent_pos[i][0] = org_x
                self.agent_pos[i][1] = org_y
                self.grid_counts[i][self.agent_pos[i][0], self.agent_pos[i][1]] += 1
                reward = 0
            elif prev_status == self.POI:
                # update the cell 
                self.grid_counts[i][self.agent_pos[i][0], self.agent_pos[i][1]] += 1
                self.grid_status[self.agent_pos[i][0], self.agent_pos[i][1]] = 1
                reward = 10
            elif prev_status == self.MAP:
                self.grid_counts[i][self.agent_pos[i][0], self.agent_pos[i][1]] += 1
                reward = 0

        # check if agents map all cells
        mapped_poi = np.count_nonzero(self.grid_status == 1)
        done = bool(mapped_poi == self.n_poi)

        # update the stuck count
        if org_x == self.agent_pos[i][0] and org_y == self.agent_pos[i][1]: # stuck
            self.stuck_counts[i] += 1
        else:
            self.stuck_counts[i] = 0

        # update the grid_agents_status
        self.grid_overlay()
        
        return self.get_agent_obs(), reward, done

    def close(self):
        pass