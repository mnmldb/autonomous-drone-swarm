class Grid(gym.Env):
    metadata = {'render.modes': ['console']}
    # action id
    XM = 0 # x minus
    XP = 1 # x plus
    YM = 2 # y minus
    YP = 3 # y plus
    
    def __init__(self, x_size=5, y_size=5, n_agents=2, fov_x=3, fov_y=3):
        super(Grid, self).__init__()
        
        # size of 2D grid
        self.x_size = x_size
        self.y_size = y_size

        # number of agents
        self.n_agents = n_agents
        self.idx_agents = list(range(n_agents)) # [0, 1, 2, ..., n_agents - 1]
        
        # initialize the mapping status
        self.init_grid()

        # initialize the position of the agent
        self.init_agent()
        
        # define action space
        n_actions = 4 # LEFT, RIGHT, TOP, BOTTOM
        self.action_space = MultiAgentActionSpace([spaces.Discrete(n_actions) for _ in range(self.n_agents)])
        
        # define observation space (fielf of view)
        self.fov_x = fov_x # number of cells around the agent
        self.fov_y = fov_y # number of cells around the agent

        self.obs_low = -np.ones(4) * 2 # low -2: out of the grid
        self.obs_high = np.ones(4) # high 1: visited
        self.observation_space = MultiAgentObservationSpace([spaces.Box(self.obs_low, self.obs_high) for _ in range(self.n_agents)])
    
    def init_agent(self, initial_pos=None):
        self.agent_pos = []
        if initial_pos is not None:
            self.agent_pos = initial_pos
            for i in range(self.n_agents):
                self.grid_status[self.agent_pos[i][0], self.agent_pos[i][1]] = 1
        else:
            for i in range(self.n_agents):
                agent_pos_x = random.randrange(0, self.x_size)
                agent_pos_y = random.randrange(0, self.x_size)
                self.agent_pos.append([agent_pos_x, agent_pos_y])
                self.grid_status[self.agent_pos[i][0], self.agent_pos[i][1]] = 1

        # iniqialize the stuck count
        self.stuck_counts = [0] * self.n_agents

    def init_grid(self):
        # initialize the mapping status
        ## -2: out of the grid
        ## -1: obstacle
        ## 0: POI that is not mapped
        ## 1: POI that is mapped
        self.grid_status = np.zeros([self.x_size, self.y_size])
        self.grid_counts = np.zeros([self.x_size, self.y_size])

        ## randomly set obstacles
        # n_obstacle = random.randrange(0, self.x_size * self.x_size * 0.2) # at most 20% of the grid
        n_obstacle = 0
        for i in range(n_obstacle):
            x_obstacle = random.randrange(1, self.x_size - 1)
            y_obstacle = random.randrange(1, self.y_size - 1)
            self.grid_status[x_obstacle, y_obstacle] = - 1
            self.grid_counts[x_obstacle, y_obstacle] = - 1
        
        # number of POI in the environment (0)
        self.n_poi = self.x_size * self.y_size - np.count_nonzero(self.grid_status)
    
    def get_coverage(self):
        mapped_poi = (self.grid_status == 1).sum()
        return mapped_poi / self.n_poi

    def get_agent_obs(self):
        self.agent_obs = []

        # observation for each agent
        for agent in range(self.n_agents):
            # default: out of the grid
            single_obs = -np.ones([self.fov_x, self.fov_y]) * 2
            for i in range(self.fov_x): # 0, 1, 2
                for j in range(self.fov_y): # 0, 1, 2
                    obs_x = self.agent_pos[agent][0] + (i - 1) # -1, 0, 1
                    obs_y = self.agent_pos[agent][1] + (j - 1) # -1, 0, 1
                    if obs_x >= 0 and obs_y >= 0 and obs_x <= self.x_size - 1 and obs_y <= self.y_size - 1:
                        single_obs[i][j] = copy.deepcopy(self.grid_status[obs_x][obs_y])
            single_obs_flat = single_obs.flatten() # convert matrix to list
            # extract the necessary cells
            xm = single_obs_flat[1]
            xp = single_obs_flat[7]
            ym = single_obs_flat[3]
            yp = single_obs_flat[5]
            single_obs_flat = np.array([xm, xp, ym, yp])
            self.agent_obs.append(single_obs_flat)
        return self.agent_obs

    def reset(self, initial_pos=None):
        # initialize the mapping status
        self.init_grid()
        # initialize the position of the agent
        self.init_agent(initial_pos)
        
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
        
    def step(self, action, i): # i: index of the drone
        # original position
        org_x  = copy.deepcopy(self.agent_pos[i][0])
        org_y  = copy.deepcopy(self.agent_pos[i][1])

        # move the agent
        if action == self.XM:
            self.agent_pos[i][0] -= 1
        elif action == self.XP:
            self.agent_pos[i][0] += 1
        elif action == self.YM:
            self.agent_pos[i][1] -= 1
        elif action == self.YP:
            self.agent_pos[i][1] += 1
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))
        
        # account for the boundaries of the grid (-2: out of the grid)
        if self.agent_pos[i][0] > self.x_size - 1 or self.agent_pos[i][0] < 0 or self.agent_pos[i][1] > self.y_size - 1 or self.agent_pos[i][1] < 0:
            self.agent_pos[i][0] = org_x
            self.agent_pos[i][1] = org_y 
            self.grid_counts[self.agent_pos[i][0], self.agent_pos[i][1]] += 1
            reward = 0
        else:
            # previous status of the cell
            prev_status = self.grid_status[self.agent_pos[i][0], self.agent_pos[i][1]]
            if prev_status == -1: # the new position is on the obstacle
                # go back to the original position
                self.agent_pos[i][0] = org_x
                self.agent_pos[i][1] = org_y
                self.grid_counts[self.agent_pos[i][0], self.agent_pos[i][1]] += 1
                reward = 0
            elif prev_status == 0:
                self.grid_counts[self.agent_pos[i][0], self.agent_pos[i][1]] += 1
                self.grid_status[self.agent_pos[i][0], self.agent_pos[i][1]] = 1
                reward = 10
            elif prev_status == 1:
                self.grid_counts[self.agent_pos[i][0], self.agent_pos[i][1]] += 1
                reward = 0

        # update the stuck count
        if org_x == self.agent_pos[i][0] and org_y == self.agent_pos[i][1]: # stuck
            self.stuck_counts[i] += 1
        else:
            self.stuck_counts[i] = 0

        # are we map all cells?
        mapped_poi = (self.grid_status == 1).sum()
        done = bool(mapped_poi == self.n_poi)
        
        return self.get_agent_obs(), reward, done

    def close(self):
        pass