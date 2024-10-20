import torch
from typing import Optional

class JointFailure:
    def __init__(self, failure_type:str ='none', dof_num:int = 0,
                failure_prob:float = 0.1, num_envs:int = 1024, max_ep_len:int = 500, device:str = "cuda:0"):
        """Class implementation for single joint failure."""

        accepted_types = ['none', 'complete', 'intermittent', 'fails_midway', 'works_midway']
        assert (failure_type in accepted_types), "Choose joint failure types from: " + str(accepted_types)
        
        self.failure_type   = failure_type
        self.dof_num        = dof_num
        self.failure_prob   = failure_prob
        self.num_envs       = num_envs
        self.max_ep_len     = max_ep_len
        self.fail_time      = torch.randint(0,self.max_ep_len,(self.num_envs,),device=device)
        self.device         = device
    
    def reset(self, env_ids: torch.Tensor, failure_prob: Optional[float] = None):
        if type(failure_prob) is float:
            self.failure_prob = failure_prob
        self.fail_time[env_ids] = torch.randint(0,self.max_ep_len,(len(env_ids),)).to(self.device)

        # print('Env has failure: ' + str(self.fail_time < (self.max_ep_len * 0.5)))
    
    def apply(self, current_dofs, targets_dofs, current_step):
        """ Method to apply joint failures.
            In case of failure, target joint state is set to the current joint state .
        """
        # Joint never works with 50% probability of episode selection
        if self.failure_type == 'complete':
            select_envs = self.fail_time < (self.max_ep_len * 0.5)
            targets_dofs[select_envs,self.dof_num] = current_dofs[select_envs,self.dof_num]
        
        # Joint fails based on input probablility
        elif (self.failure_type == 'intermittent'):
            select_envs = self.failure_prob > torch.rand(targets_dofs.shape[0]).to(self.device)
            targets_dofs[select_envs,self.dof_num] = current_dofs[select_envs,self.dof_num]
        
        # Joint stops working after randomly determined step, fail_time set at episode reset
        elif (self.failure_type == 'fails_midway'):
            select_envs = current_step >= self.fail_time
            targets_dofs[select_envs,self.dof_num] = current_dofs[select_envs,self.dof_num]

        # Joint starts working after randomly determined step, fail_time set at episode reset
        elif (self.failure_type == 'works_midway'):
            select_envs = current_step < self.fail_time
            targets_dofs[select_envs,self.dof_num] = current_dofs[select_envs,self.dof_num]
        

class MultiJointFailure:
        def __init__(self):
            pass