import torch
import numpy as np
from typing import Optional

## Functionality
# 1) Ability to update joint list
# 2) Ability to randomize joint selection across all reseting environments (including default no joint 7)
# 3) Ability to select from failure types
# 4) Ability to apply failure specific randomizations
# 5) Track statistics: percentage of joint selection. Reset at curriculum change

# Updates
# Joint 7 is a valid curriculum entry
# set_dof_id_list does not append 7 anymore
# apply remains the same 


class JointFailure:
    def __init__(self, failure_type:str ='none', dof_id_list:list = [0],
                failure_prob:float = 0.1, num_envs:int = 1024, max_ep_len:int = 500, device:str = "cuda:0"):
        """Class implementation for single joint failure."""

        accepted_types = ['none', 'complete', 'intermittent', 'fails_midway', 'works_midway']
        assert (failure_type in accepted_types), "Choose joint failure types from: " + str(accepted_types)
        
        self.failure_type       = failure_type          # Type of failure
        self.failure_prob       = failure_prob          # Used for intermittent failure type
        self.num_envs           = num_envs              # Number of parallel environments
        self.max_ep_len         = max_ep_len            # Max episode length
        self.set_dof_id_list(dof_id_list=dof_id_list)   # Set initial dof_id_list

        # Initialize dof_ids tensor. It contains random choice of dof failure per environment
        self.dof_ids = torch.from_numpy(
            np.random.choice(self.dof_id_list,self.num_envs).astype(int)
            ).to(device=device)
        # Initialization of failure time for fail during runs
        self.fail_time          = torch.randint(0,self.max_ep_len,(self.num_envs,), device=device)
        # Joint failures applied. 7 implies no failure. Updated at each apply() call
        self.envs_joints_failed = torch.zeros(num_envs, device=device, dtype=int) + 7
        self.device             = device
        
    
    def reset(self, env_ids: torch.Tensor, failure_prob: Optional[float] = None):
        """Updates values for resetting environments"""

        if type(failure_prob) is float:
            self.failure_prob = failure_prob
        
        self.dof_ids[env_ids] = torch.from_numpy(
            np.random.choice(self.dof_id_list,len(env_ids)).astype(int)
            ).to(device=self.device)
        
        self.fail_time[env_ids] = torch.randint(0,self.max_ep_len,(len(env_ids),)).to(self.device)


    
    def set_dof_id_list(self, dof_id_list:list = [0]):
        """Used to update curriculum"""

        assert all(i < 8 for i in dof_id_list), f"Values in dof_id_list must be integers between 0-7, where 7 implies no selection"
        self.dof_id_list = dof_id_list.copy()
        # self.env_selection_prob = 1 / len(self.dof_id_list)
    
    def apply(self, current_dofs, targets_dofs, current_step):
        """ Method to apply joint failures.
            In case of failure, target joint state is set to the current joint state .
        """
        
        # Reset joint failure info
        self.envs_joints_failed[:] = 7

        # Joint never works with some probability of episode selection
        if self.failure_type == 'complete':
            select_envs = self.dof_ids != 7
            select_dof_ids = self.dof_ids[select_envs]
            # print('Failing env: ' + str(select_envs) + ', Failing joint: ' + str(select_dof_ids))
            targets_dofs[select_envs,select_dof_ids] = current_dofs[select_envs,select_dof_ids]
            self.envs_joints_failed[select_envs] = select_dof_ids
        
        # Joint fails based on input probablility
        elif (self.failure_type == 'intermittent'):
            select_envs = (
                (self.dof_ids != 7)
                and (self.failure_prob > torch.rand(targets_dofs.shape[0]).to(self.device))
                )
            select_dof_ids = self.dof_ids[select_envs]
            targets_dofs[select_envs,select_dof_ids] = current_dofs[select_envs,select_dof_ids]
            self.envs_joints_failed[select_envs] = select_dof_ids
        
        # Joint stops working after randomly determined step, fail_time set at episode reset
        elif (self.failure_type == 'fails_midway'):
            select_envs = (
                (self.dof_ids != 7)
                and (current_step >= self.fail_time)
                )
            select_dof_ids = self.dof_ids[select_envs]
            targets_dofs[select_envs,select_dof_ids] = current_dofs[select_envs,select_dof_ids]
            self.envs_joints_failed[select_envs] = select_dof_ids

        # Joint starts working after randomly determined step, fail_time set at episode reset
        elif (self.failure_type == 'works_midway'):
            select_envs = (
                (self.dof_ids != 7)
                and (current_step < self.fail_time)
                )
            select_dof_ids = self.dof_ids[select_envs]
            targets_dofs[select_envs,select_dof_ids] = current_dofs[select_envs,select_dof_ids]
            self.envs_joints_failed[select_envs] = select_dof_ids
    
    def get_env_joint_failure_indices(self):
        """This method returns the joint indices failed for each environment.
        Environment with no joint failueres have a value of 7.
        The variable is updated in the self.apply(...) function at each step.
        """
        return self.envs_joints_failed
        
class JointFailureStatsWrapper(JointFailure):
    def __init__(self, failure_type = 'none', dof_id_list = [0], failure_prob = 0.1, num_envs = 1024, max_ep_len = 500, device = "cuda:0"):
        self.sum_values = torch.zeros(len(dof_id_list), device=device)
        print('Joint failure has been initialized')
        # print('Percentages at initialization: ' + str(self.sum_values/(len(dof_id_list)+1)))
        self.num_of_resets = 0
        self.device = device

        super().__init__(failure_type, dof_id_list, failure_prob, num_envs, max_ep_len, device)
        # initialize sum logging variable [num joints] - Based on initial curriculum
    
    def reset(self, env_ids, failure_prob = None):
        super().reset(env_ids, failure_prob)

        self.num_of_resets += len(env_ids)
        # Update sum for env_id by +1
        for i in range(len(env_ids)):
            for j in range(len(self.dof_id_list)):
                if self.dof_id_list[j] == self.dof_ids[i]:
                    self.sum_values[j] += 1.0
    
    def set_dof_id_list(self, dof_id_list = [0]):
        # Print stats for completed curriculums
        if self.num_of_resets != 0:
            print('Curriculum Percentages: ' + str(self.sum_values/self.num_of_resets))
            print('Total: ' + str(torch.sum(self.sum_values/self.num_of_resets)))
            self.num_of_resets = 0

        super().set_dof_id_list(dof_id_list)
        
        # Reset sum variable
        self.sum_values = torch.zeros(len(self.dof_id_list), device=self.device)

class MultiJointFailure:
        def __init__(self):
            pass