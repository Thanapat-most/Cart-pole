import gym
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
#แสดงจากlist เป็นตาราง
class Agent():
    def __init__(self):
        n_bins = 10
        self.car_bins = pd.cut([-2,2],bins=n_bins,retbins=True)[1][1:-1]
        self.velocity_bins = pd.cut([-3,3],bins=n_bins,retbins=True)[1][1:-1]
        self.pole_bins = pd.cut([-0.3,0.3],bins=n_bins,retbins=True)[1][1:-1]
        self.omega_bins = pd.cut([-3,3],bins=n_bins,retbins=True)[1][1:-1]
        self.q_table=np.zeros((n_bins,n_bins,n_bins,n_bins)+(2,))
        self.learning_rate=0.05
        self.discount_factor=0.95
        self.epsilon=1.0
        self.decay_factor=0.999
        self.total_reward=[]
        self.total_reward_test = []
    def play(self,env,number_of_episode=8000,isRender = False):
        for i in range(number_of_episode):
            print("episode {} of {}".format(i+1,number_of_episode))
            observation=env.reset()
            car,velocity,pole,omega = observation
            state = (self.digitize(car,self.car_bins),self.digitize(velocity,self.velocity_bins),self.digitize(pole,self.pole_bins),self.digitize(omega,self.omega_bins))
            self.epsilon*=self.decay_factor
            total_reward=0
            end_game = False
            while not end_game:
                if isRender:
                    env.render()
                if self.q_table_empty(state) or self.prob(self.epsilon):
                    action= self.action_by_random(env)
                else:
                    action= self.get_action_with_highreward(state)
               
                
                new_observation,reward,end_game,_ = env.step(action)
                if end_game:
                    reward=-200
                else:
                    total_reward+=reward
                new_car,new_velocity,new_pole,new_omega = new_observation
                new_state =(self.digitize(new_car,self.car_bins),self.digitize(new_velocity,self.velocity_bins),self.digitize(new_pole,self.pole_bins),self.digitize(new_omega,self.omega_bins))
                #q table
                self.q_table[state][action]+=self.learning_rate*(reward+self.discount_factor*self.getreward(new_state)-self.q_table[state][action])
                #total_reward +=reward
                state = new_state
            self.total_reward.append(total_reward)
            #print(tabulate(self.q_table,showindex="always",headers=["State","Action 0(Forward 1 step)","Action 1(Back to 0)"]))
    '''
    def test(self,env,number_of_episode=1):
        for i in range(number_of_episode):
            print("episode {} of {}".format(i,number_of_episode))
            observation=env.reset()
            car,velocity,pole,omega = observation
            state = (self.digitize(car,self.car_bins),self.digitize(velocity,self.velocity_bins),self.digitize(pole,self.pole_bins),self.digitize(omega,self.omega_bins))
            
            total_reward=0
            end_game = False
            while not end_game:
                action= self.get_action_with_highreward(state)
            
                new_observation,reward,end_game,_ = env.step(action)
                
                total_reward +=reward

            
            
                
                new_car,new_velocity,new_pole,new_omega = new_observation
                new_state =(self.digitize(new_car,self.car_bins),self.digitize(new_velocity,self.velocity_bins),self.digitize(new_pole,self.pole_bins),self.digitize(new_omega,self.omega_bins))
                state = new_state
                
            self.total_reward_test.append(total_reward)
'''
    
    def q_table_empty(self,state):
        return np.sum(self.q_table[state])==0
    def action_by_random(self,env):
        return env.action_space.sample()
    def get_action_with_highreward(self,state):
        return np.argmax(self.q_table[state])
    def getreward(self,state):
        return np.max(self.q_table[state])
    def prob(self,prob):
        return np.random.random()< prob
    def digitize(self,value,bins):
        return np.digitize(x=value,bins=bins)

env=gym.make('CartPole-v1')
agent=Agent()
agent.play(env)
agent.play(env,number_of_episode=3,isRender = True)
plt.plot(agent.total_reward)
agent.test(env)

plt.show()
