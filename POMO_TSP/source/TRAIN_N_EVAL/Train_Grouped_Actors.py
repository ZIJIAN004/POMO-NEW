
"""
The MIT License

Copyright (c) 2020 Yeong-Dae Kwon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np
import time

# For debugging
from IPython.core.debugger import set_trace

# Hyper Parameters
from HYPER_PARAMS import *
from TORCH_OBJECTS import *

from source.utilities import Average_Meter
from source.travelling_saleman_problem import TSP_DATA_LOADER__RANDOM, GROUP_ENVIRONMENT


########################################
# TRAIN
########################################

def TRAIN(actor_group, epoch, timer_start, logger):

    actor_group.train()

    train_loader = TSP_DATA_LOADER__RANDOM(num_sample=TRAIN_DATASET_SIZE, num_nodes=TSP_SIZE, batch_size=TRAIN_BATCH_SIZE)
    episode=0
    logger_start = time.time()
    for data,ref in train_loader:
        # data.shape = (batch_s, TSP_SIZE, 2)            #需要修改，需要先修改数据集，变成TSP_size或者TSP_size+1
        # ref.shape= (batch_s, TSP_SIZE+1)
        
        batch_s = data.size(0)
        episode = episode + batch_s

        # Actor Group Move                               #整段逻辑需要修改，需要先训练从半途开始，然后再训练从头开始，这一段循环只对一个问题实例做一次采样然后训练，因为有个外部循环会多次执行他们
        ###############################################
        env = GROUP_ENVIRONMENT(data)                    #利用ref去构建每一步该走的select node mat，这个维度应该是batchsize，groupsize，其中每一行的列数值应该完全相同
        group_s = TSP_SIZE
        group_s_p=TSP_SIZE-(ref.size(1)//2)                               #group_s也要有两个，因为一个是部分解用的，一个是全部用的

        #Training the model using the partial best solution
        ###############################################
        group_state_p, reward_p, done_p = env.reset(group_size=group_s_p)  
        actor_group.reset(group_state_p)     
        #Push the model to the state of the partial solution
        for i in range(ref.size(1)//2):
            col = ref[:, i] 
            col_unsqueezed = col.unsqueeze(1)
            pre_action = col_unsqueezed.expand(-1, group_s_p)
            group_state_p, reward_p, done_p = env.step(pre_action)
    
        # First Move is given
        first_action = ref[:, (ref.size(1)//2 + 1) : -1] #这个firstmove的逻辑就需要修改了
        group_state_p, reward_p, done_p = env.step(first_action)

        group_prob_list = Tensor(np.zeros((batch_s, group_s_p, 0)))
        while not done_p:
            actor_group.update(group_state_p)
            action_probs = actor_group.get_action_probabilities()
            # shape = (batch, group, TSP_SIZE)
            action = action_probs.reshape(batch_s*group_s_p, -1).multinomial(1).squeeze(dim=1).reshape(batch_s, group_s_p)
            # shape = (batch, group)
            group_state_p, reward_p, done_p = env.step(action)

            batch_idx_mat = torch.arange(batch_s)[:, None].expand(batch_s, group_s_p)
            group_idx_mat = torch.arange(group_s_p)[None, :].expand(batch_s, group_s_p)
            chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, action].reshape(batch_s, group_s_p)
            # shape = (batch, group)
            group_prob_list = torch.cat((group_prob_list, chosen_action_prob[:, :, None]), dim=2)

        # LEARNING - Actor
        ###############################################
        group_reward = reward_p
        group_log_prob = group_prob_list.log().sum(dim=2)
        # shape = (batch, group)

        group_advantage = group_reward - group_reward.mean(dim=1, keepdim=True)

        group_loss = -group_advantage * group_log_prob
        # shape = (batch, group)
        loss = group_loss.mean()

        actor_group.optimizer.zero_grad()
        loss.backward()
        actor_group.optimizer.step()

        #reset the counting logic
        distance_AM = Average_Meter()
        actor_loss_AM = Average_Meter()
        episode = 0

        #Train the model using the whole RL
        ###############################################
    for data,_ in train_loader:
        # data.shape = (batch_s, TSP_SIZE, 2)

        batch_s = data.size(0)

        # Actor Group Move
        ###############################################
        env = GROUP_ENVIRONMENT(data)
        group_state, reward, done = env.reset(group_size=group_s)
        actor_group.reset(group_state)

        # First Move is given
        first_action = LongTensor(np.arange(group_s))[None, :].expand(batch_s, group_s)
        group_state, reward, done = env.step(first_action)

        group_prob_list = Tensor(np.zeros((batch_s, group_s, 0)))
        while not done:
            actor_group.update(group_state)
            action_probs = actor_group.get_action_probabilities()
            # shape = (batch, group, TSP_SIZE)
            action = action_probs.reshape(batch_s*group_s, -1).multinomial(1).squeeze(dim=1).reshape(batch_s, group_s)
            # shape = (batch, group)
            group_state, reward, done = env.step(action)

            batch_idx_mat = torch.arange(batch_s)[:, None].expand(batch_s, group_s)
            group_idx_mat = torch.arange(group_s)[None, :].expand(batch_s, group_s)
            chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, action].reshape(batch_s, group_s)
            # shape = (batch, group)
            group_prob_list = torch.cat((group_prob_list, chosen_action_prob[:, :, None]), dim=2)

        # LEARNING - Actor
        ###############################################
        group_reward = reward
        group_log_prob = group_prob_list.log().sum(dim=2)
        # shape = (batch, group)

        group_advantage = group_reward - group_reward.mean(dim=1, keepdim=True)

        group_loss = -group_advantage * group_log_prob
        # shape = (batch, group)
        loss = group_loss.mean()

        actor_group.optimizer.zero_grad()
        loss.backward()
        actor_group.optimizer.step()

        # RECORDING
        ###############################################
        max_reward, _ = group_reward.max(dim=1)
        distance_AM.push(-max_reward)  # reward was given as negative dist
        actor_loss_AM.push(group_loss.detach().reshape(-1))

        # LOGGING
        ###############################################
        if (time.time()-logger_start > LOG_PERIOD_SEC) or (episode == TRAIN_DATASET_SIZE):
            timestr = time.strftime("%H:%M:%S", time.gmtime(time.time()-timer_start))
            log_str = 'Ep:{:03d}-{:07d}({:5.1f}%)  T:{:s}  ALoss:{:+5f}  CLoss:{:5f}  Avg.dist:{:5f}' \
                .format(epoch, episode, episode/TRAIN_DATASET_SIZE*100,
                        timestr, actor_loss_AM.result(), 0,
                        distance_AM.result())
            logger.info(log_str)
            logger_start = time.time()

    # LR STEP, after each epoch
    actor_group.lr_stepper.step()




