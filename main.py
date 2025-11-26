import gymnasium as gym
import numpy as np
env = gym.make("MountainCar-v0",render_mode="rgb_array")


DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

LEARNING_RATE= 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY =2000
epsilon = 0.5
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.01


def get_discrete_state(state):
    discrete_state = (state- env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(int))



for episode in range(EPISODES):
    
    done = False 
    state, info = env.reset()
    state = get_discrete_state(state)
    
    while (not done):
        
        if np.random.random() > epsilon:
            action = np.argmax(q_table[state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, terminated,truncated, info = env.step(action)
        new_state = get_discrete_state(new_state)
       
        done = (terminated or truncated)
        
        if not done:
            max_future_q = np.max(q_table[new_state])
            current_q = q_table[state + (action,)]

            new_q = (1-LEARNING_RATE)*current_q + LEARNING_RATE*(reward + DISCOUNT*max_future_q)
            q_table[state + (action,)] = new_q
        elif terminated:
            print(f"We made it on episode {episode}")
            q_table[state + (action,)] =0 

        state = new_state
    epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
        

    

env.close()

