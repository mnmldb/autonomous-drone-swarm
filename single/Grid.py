class Grid(gym.Env):
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
    
    def __init__(self, x_size=5, y_size=5, fov_x=3, fov_y=3):
        super(Grid, self).__init__()
        
        # size of 2D grid
        self.x_size = x_size
        self.y_size = y_size
        
        # initialize the position of the agent
        self.init_agent()
        
        # initialize the mapping status
        self.init_grid()
        
        # define action space
        n_actions = 8 # LEFT, RIGHT, TOP, BOTTOM
        self.action_space = spaces.Discrete(n_actions)
        
        # define observation space (fielf of view)
        self.fov_x = fov_x # number of cells around the agent
        self.fov_y = fov_y # number of cells around the agent
        self.obs_low = -np.ones(self.fov_x * self.fov_y) * 2
        self.obs_high = np.ones(self.fov_x * self.fov_y)  # -2: out of the grid, -1: obstacle, 0: not visited, 1: visited
        self.observation_space = spaces.Box(self.obs_low, self.obs_high)
    
    def init_agent(self):
        # initialize the agent position
        agent_pos_x = random.randrange(0, self.x_size)
        agent_pos_y = random.randrange(0, self.y_size)
        self.agent_pos = [agent_pos_x, agent_pos_y]

    def init_grid(self):
        # initialize the mapping status
        ## -2: out of the grid
        ## -1: obstacle
        ## 0: POI that is not mapped
        ## 1: POI that is mapped
        self.grid_status = np.zeros([self.x_size, self.y_size])
        self.grid_counts = np.zeros([self.x_size, self.y_size])

        ## randomly set obstacles
        n_obstacle = random.randrange(0, self.x_size * self.x_size * 0.2) # at most 20% of the grid
        for i in range(n_obstacle):
            x_obstacle = random.randrange(1, self.x_size - 1)
            y_obstacle = random.randrange(1, self.y_size - 1)
            self.grid_status[x_obstacle, y_obstacle] = - 1
            self.grid_counts[x_obstacle, y_obstacle] = - 1
        
        # number of POI in the environment (0)
        self.n_poi = self.x_size * self.y_size - np.count_nonzero(self.grid_status)
    
    def get_coverage(self):
        mapped_poi = np.count_nonzero(self.grid_status == 1) + np.count_nonzero(self.grid_status == 2)
        return mapped_poi / self.n_poi

    def get_agent_obs(self):
        # default: out of the grid
        agent_obs_mat = -np.ones([self.fov_x, self.fov_y]) * 2
        for i in range(self.fov_x): # 0, 1, 2
            for j in range(self.fov_y): # 0, 1, 2
                obs_x = self.agent_pos[0] + (i - 1) # -1, 0, 1
                obs_y = self.agent_pos[1] + (j - 1) # -1, 0, 1
                if obs_x >= 0 and obs_y >= 0 and obs_x <= self.x_size - 1 and obs_y <= self.y_size - 1:
                    agent_obs_mat[i][j] = self.grid_status[obs_x][obs_y]
        self.agent_obs = agent_obs_mat.flatten() # convert matrix to list
        return self.agent_obs

    def reset(self):
        # initialize the position of the agent
        self.init_agent()
        # initialize the mapping status
        self.init_grid()
        return self.get_agent_obs()
        
    def step(self, action):
        # original position
        org_x  = self.agent_pos[0]
        org_y  = self.agent_pos[1]

        # move the agent
        if action == self.XM: # 0
            self.agent_pos[0] -= 1
        elif action == self.XP: # 1
            self.agent_pos[0] += 1
        elif action == self.YM: # 2
            self.agent_pos[1] -= 1
        elif action == self.YP: # 3
            self.agent_pos[1] += 1
        elif action == self.XMYM: # 4
            self.agent_pos[0] -= 1
            self.agent_pos[1] -= 1
        elif action == self.XMYP: # 5
            self.agent_pos[0] -= 1
            self.agent_pos[1] += 1
        elif action == self.XPYM: # 6
            self.agent_pos[0] += 1
            self.agent_pos[1] -= 1
        elif action == self.XPYP: # 7
            self.agent_pos[0] += 1
            self.agent_pos[1] += 1
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))
        
        # account for the boundaris of the grid (-2: out of the grid)
        if self.agent_pos[0] > self.x_size - 1 or self.agent_pos[0] < 0 or self.agent_pos[1] > self.y_size - 1 or self.agent_pos[1] < 0:
            self.agent_pos[0] = org_x
            self.agent_pos[1] = org_y 
            self.grid_counts[self.agent_pos[0], self.agent_pos[1]] += 1
            reward = -10
        else:
            # previous status of the cell
            prev_status = self.grid_status[self.agent_pos[0], self.agent_pos[1]]
            if prev_status == -1: # the new position is on the obstacle
                # go back to the original position
                self.agent_pos[0] = org_x
                self.agent_pos[1] = org_y
                self.grid_counts[self.agent_pos[0], self.agent_pos[1]] += 1
                reward = -10
            elif prev_status == 0:
                # update the cell 
                self.grid_status[self.agent_pos[0], self.agent_pos[1]] = 1
                self.grid_counts[self.agent_pos[0], self.agent_pos[1]] += 1
                reward = 10
            elif prev_status == 1:
                self.grid_counts[self.agent_pos[0], self.agent_pos[1]] += 1
                reward = -1
    
        # are we map all cells?
        mapped_poi = np.count_nonzero(self.grid_status == 1) + np.count_nonzero(self.grid_status == 2)
        done = bool(mapped_poi == self.n_poi)

        # optionally we can pass additional info
        info = {}
        
        return self.get_agent_obs(), reward, done, info

    def close(self):
        pass