# MIT 许可证

# 版权所有 (c) 2025 ReinFlow Authors

# 特此免费授予任何获得本软件及相关文档文件（以下简称“软件”）副本的人，
# 不受限制地处理本软件的权利，包括但不限于使用、复制、修改、合并、发布、
# 分发、再许可和/或销售本软件副本的权利，并允许向其提供本软件的人这样做，
# 但须符合以下条件：

# 上述版权声明和本许可声明应包含在本软件的所有副本或主要部分中。

# 本软件按“原样”提供，不提供任何明示或暗示担保，包括但不限于适销性、
# 特定用途适用性和不侵权担保。无论在合同诉讼、侵权诉讼或其他诉讼中，
# 对于因本软件或本软件的使用或其他处理而产生、由此引发或与之相关的
# 任何索赔、损害或其他责任，作者或版权持有人均不承担责任。


"""
微调训练。
"""
import os
import logging
log = logging.getLogger(__name__)
from tqdm import tqdm as tqdm
import numpy as np
import torch
from agent.finetune.reinflow.train_ppo_agent import TrainPPOAgent
from model.flow.ft_ppo.ppoflow import PPOFlow
from agent.finetune.reinflow.buffer import PPOFlowBuffer#, PPOFlowBufferGPU
from util.scheduler_simple import get_scheduler
import matplotlib.pyplot as plt
# 可在 CPU 或 CUDA 上定义 buffer。当前 GPU 版本加速效果不明显...
# 通信可能是瓶颈，目前仅将 GPU 瞬时利用率从 7% 提升到 13%
# 这可能是因为 mujoco 在 CPU 上产数据，而我们频繁在 CPU/GPU 间搬运数据。


# 该脚本同时适用于预训练 1-ReFlow 与 ShortCutFlows。



class TrainPPOFlowAgent(TrainPPOAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        # 奖励时域：当前默认始终设为 act_steps
        self.skip_initial_eval=cfg.get('skip_initial_eval', False)
        self.reward_horizon = cfg.get("reward_horizon", self.act_steps)
        self.inference_steps = self.model.inference_steps
        self.ft_denoising_steps = self.model.ft_denoising_steps
        self.repeat_samples = cfg.train.get("repeat_samples", False)
        
        self.normalize_act_space_dim = True   # 在时域步和动作维上归一化熵与 logprob，环境规模增大时无需额外调整熵系数。
        self.normalize_denoising_horizon = True     # 计算单次模拟动作对应马尔可夫链 logprob 时，对去噪时域做归一化。
        self.lr_schedule = cfg.train.lr_schedule
        self.clip_intermediate_actions = cfg.train.get("clip_intermediate_actions", True)
        self.account_for_initial_stochasticity = cfg.train.get('account_for_initial_stochasticity', True)
        if self.lr_schedule not in ["fixed", "adaptive_kl"]:
            raise ValueError("lr_schedule should be 'fixed' or 'adaptive_kl'")
        self.actor_lr = cfg.train.actor_lr
        self.critic_lr = cfg.train.critic_lr
        
        self.model: PPOFlow

        if self.model.noise_scheduler_type == 'const_schedule_itr':
            self.explore_noise_scheduler = get_scheduler(schedule_type='cosine_warmup',
                                                            min=0.016,
                                                            warmup_steps=self.n_train_itr * 0.01,
                                                            max=0.08, # 0.15,
                                                            hold_steps=self.n_train_itr * 0.29,
                                                            anneal_steps=self.n_train_itr * 0.7)
            
            explore_noises = [self.explore_noise_scheduler(t) for t in np.arange(self.n_train_itr)]
            plt.figure()
            plt.plot(np.arange(self.n_train_itr), explore_noises)
            name=os.path.join(self.logdir,'explore_noise')+'.png'
            plt.savefig(name)
            plt.close()
            log.info("Exploration noise saved to %s" % name)
        elif self.model.noise_scheduler_type == 'learn_decay':
            max_std=cfg.model.max_logprob_denoising_std
            min_std=cfg.model.min_logprob_denoising_std
            self.max_noise_decay_ratio=cfg.train.get('max_noise_decay_ratio', 0.7)
            max_std_decayed=min_std*(1-self.max_noise_decay_ratio)+max_std*self.max_noise_decay_ratio
            # min_std*0.20+max_std*0.80
            # min_std*0.4+max_std*0.6
            # cfg.model.min_logprob_denoising_std*1.5
            self.max_noise_hold_ratio=cfg.train.get('max_noise_hold_ratio', 0.35)
            self.explore_noise_scheduler = get_scheduler(schedule_type='cosine',
                                                            max=max_std,
                                                            hold_steps=self.n_train_itr * self.max_noise_hold_ratio,
                                                            anneal_steps=self.n_train_itr * (1-self.max_noise_hold_ratio),
                                                            min=max_std_decayed)
            max_explore_noises = [self.explore_noise_scheduler(t) for t in np.arange(self.n_train_itr)]
            min_explore_noises = [min_std for _ in np.arange(self.n_train_itr)]
            plt.figure()
            plt.plot(np.arange(self.n_train_itr), max_explore_noises, label=f'max_std:{max_std:.2f} to {max_std_decayed:.2f}')
            plt.plot(np.arange(self.n_train_itr), min_explore_noises, label=f'min_std:{min_std:.2f}')
            plt.legend()
            name=os.path.join(self.logdir,'explore_noise')+'.png'
            plt.savefig(name)
            plt.close()
            log.info("Exploration noise level bounds saved to %s" % name)
        else:
            max_std=cfg.model.max_logprob_denoising_std
            min_std=cfg.model.min_logprob_denoising_std
            max_explore_noises = [max_std for _ in np.arange(self.n_train_itr)]
            min_explore_noises = [min_std for _ in np.arange(self.n_train_itr)]
            log.info(f"Received self.model.noise_scheduler_type={self.model.noise_scheduler_type}, will use constant noise ranges [{min_std:.2f}, {max_std:.2f}]")
            plt.figure()
            plt.plot(np.arange(self.n_train_itr), max_explore_noises, label=f'max_std:{max_std:.2f}')
            plt.plot(np.arange(self.n_train_itr), min_explore_noises, label=f'min_std:{min_std:.2f}')
            plt.legend()
            name=os.path.join(self.logdir,'explore_noise')+'.png'
            plt.savefig(name)
            plt.close()
            log.info("Exploration noise level bounds saved to %s" % name)
    
        self.initial_ratio_error_threshold = 1e-6 # 对于基于 state 且无数据增强的任务，batch=0、epoch=0 时 logprob ratio 应严格为 1.00。

    def init_buffer(self):
        self.buffer = PPOFlowBuffer(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            n_ft_denoising_steps= self.inference_steps, 
            horizon_steps=self.horizon_steps,
            act_steps=self.act_steps,
            action_dim=self.action_dim,
            n_cond_step=self.n_cond_step,
            obs_dim=self.obs_dim,
            save_full_observation=self.save_full_observations,
            furniture_sparse_reward=self.furniture_sparse_reward,
            best_reward_threshold_for_success=self.best_reward_threshold_for_success,
            reward_scale_running=self.reward_scale_running,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            reward_scale_const=self.reward_scale_const,
            device=self.device,
        ) 
    
    def resume_training(self):
        super().resume_training()
        if self.model.noise_scheduler_type == 'const':
            updated_noise_std_range=[
                self.cfg.model.min_logprob_denoising_std, 
                self.cfg.model.max_logprob_denoising_std
            ]
            self.model.actor_ft.explore_noise_net.set_noise_range(updated_noise_std_range)
            log.info(f"Updated noise_std_range={updated_noise_std_range} (self.model.noise_scheduler_type={self.model.noise_scheduler_type})")   
    
    def run(self):
        self.init_buffer()      # 创建经验池，包含obs, action, reward, done, value, logprob, chains
        self.prepare_run()
        # PPO buffer：首轮前显式 reset；后续按 step 写入时会在新迭代 step=0 处自动重置，故仅此处需一次初始 reset。
        self.buffer.reset()     # 经验池初始清空
        if self.resume:            # 恢复训练
            self.resume_training()
        while self.itr < self.n_train_itr:  # 循环训练n_train_itr轮
            self.prepare_video_path()  # 准备视频路径
            self.set_model_mode()  # 设置 train/eval 模式
            self.reset_env()    # 重置环境
            self.buffer.update_full_obs()  # 更新经验池
            # Rollout：收集 n_steps 条 transition，供本迭代 GAE / PPO 更新使用。
            for step in range(self.n_steps):    # 一轮 rollout 采样 n_steps
                with torch.no_grad():   # 不计算梯度
                    # 当前时刻并行环境里的所有观测，整理成一个 batch，送入模型
                    cond = {
                        "state": torch.tensor(self.prev_obs_venv["state"], device=self.device, dtype=torch.float32)
                    }
                    # critic 估计状态价值
                    value_venv = self.get_value(cond=cond)

                    # get_samples_logprobs（内部 model.get_actions）：
                    # 输入 cond：{"state": Tensor[B, cond_steps, obs_dim]}，B=n_envs，与并行环境 batch 对齐。
                    # 输出（默认 ret_device=cpu，均为 numpy，便于 mujoco）：
                    #   action_samples [B, horizon_steps, action_dim] — 去噪终点动作序列；
                    #   chains_venv      [B, K+1, horizon_steps, action_dim] — 每步 xt，K=inference_steps；
                    #   logprob_venv     [B] — 每个环境一条轨迹的标量 log p（尺度由下方 bool 控制）。
                    action_samples, chains_venv, logprob_venv = self.get_samples_logprobs(
                        cond=cond,
                        # 以下为 logprob 计算时的尺度与建模选项（定义见 model.get_logprobs）
                        normalize_denoising_horizon=self.normalize_denoising_horizon,  # True：总 logprob 除以参与累加的步数，减弱推理步数 K 变化带来的量级漂移
                        normalize_act_space_dimension=self.normalize_act_space_dim,  # True：再除以动作总维数，便于不同 act_dim 间调参
                        clip_intermediate_actions=self.clip_intermediate_actions,  # True：flow 每步预测均值在送入转移分布前 clamp，抑制中间态发散
                        account_for_initial_stochasticity=self.account_for_initial_stochasticity,  # True：logprob 含 x0~N(0,I) 初分布项；False 仅转移项
                    )
                
                action_venv = action_samples[:, : self.act_steps]   # 截取前 act_steps 步动作，送入环境
                # 环境步进，获取新观测、奖励、终止信号等
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
                
                # 保存完整观测信息
                self.buffer.save_full_obs(info_venv)
                # 把 rollout 数据写进 buffer
                self.buffer.add(step, self.prev_obs_venv["state"], chains_venv, reward_venv, terminated_venv, truncated_venv, value_venv, logprob_venv)
                
                self.prev_obs_venv = obs_venv  # 更新上一时刻观测
                # 训练模式下累计全局步数（环境数 × 每步动作长度）；eval 不计。
                self.cnt_train_step+= self.n_envs * self.act_steps if not self.eval_mode else 0
            self.buffer.summarize_episode_reward()  # 一轮 rollout 完成后的汇总
            
            # 如果不是 eval，就开始 PPO 更新
            if not self.eval_mode:
                # 把 rollout 原始数据加工成 PPO 真正训练用的数据集。
                self.buffer.update(obs_venv, self.model.critic)
                # 执行梯度下降
                self.agent_update(verbose=self.verbose)
            
            # self.plot_state_trajecories() # 仅 D3IL 使用
            # 日志、学习率、噪声调度、保存模型
            self.log()  # diffusion_min_sampling_std 等
            self.update_lr()
            self.adjust_finetune_schedule()  # ReFlow policy 噪声/微调调度
            self.save_model()
            self.itr += 1 
            
    def adjust_finetune_schedule(self):
        # 中间步噪声在单次 rollout 内固定，但会随训练进程变化
        if self.model.noise_scheduler_type == 'const_schedule_itr':
            explore_noise_std = self.explore_noise_scheduler(self.itr)
            self.model.actor_ft.set_logprob_noise_levels(force_level=explore_noise_std)
        
        # 逐步降低噪声上界，避免高噪声样本伤害模型。
        if self.model.noise_scheduler_type == 'learn_decay':
            updated_noise_std_range=[
                self.model.actor_ft.min_logprob_denoising_std, 
                self.explore_noise_scheduler(self.itr)
            ]
            self.model.actor_ft.explore_noise_net.set_noise_range(updated_noise_std_range)
            log.info(f"Updated noise_std_range={updated_noise_std_range} (self.model.noise_scheduler_type={self.model.noise_scheduler_type})")
        
    # 重载...
    def save_model(self, only_save_policy_network=False):
        """
        将模型保存到磁盘；由于进行的是 RLFT，因此不记录 EMA。
        评估场景可将 ``only_save_policy`` 设为 True。此时不保存 critic 和探索噪声网络，更省空间。
        继续训练场景可将 ``only_save_policy`` 设为 False。此时会保存恢复训练所需的全部内容。
        """
        policy_network_state_dict = {
            'network.'+key:value for key, value in self.model.actor_ft.policy.state_dict().items()
        } # 这是一个 FlowMLP 网络，可加载到 ReFlow 对象的 .network 属性。
        
        if only_save_policy_network:
            data = {
                "itr": self.itr,
                "cnt_train_steps": self.cnt_train_step,
                "policy": policy_network_state_dict,
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "actor_lr_scheduler": self.actor_lr_scheduler.state_dict(),
                "critic_lr_scheduler": self.critic_lr_scheduler.state_dict(),
            }
        else:
            data = {
                "itr": self.itr,
                "cnt_train_steps": self.cnt_train_step,
                "model": self.model.state_dict(),  # 用于恢复训练
                "policy": policy_network_state_dict,  # 用于评估的 flow policy，不含 critic 和探索噪声网络
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "actor_lr_scheduler": self.actor_lr_scheduler.state_dict(),
                "critic_lr_scheduler": self.critic_lr_scheduler.state_dict(),
            }
        
        # 始终保存 last 模型，便于恢复训练。
        save_path = os.path.join(self.checkpoint_dir,f"last.pt")
        torch.save(data, os.path.join(self.checkpoint_dir, save_path))
        
        # 按需保存中间模型
        if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
            save_path = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
            torch.save(data, os.path.join(self.checkpoint_dir, save_path))
            log.info(f"\n Saved model at itr={self.itr} to {save_path}\n ")
        
        # 保存截至当前评估最优模型
        if self.is_best_so_far:
            save_path = os.path.join(self.checkpoint_dir,f"best.pt")
            torch.save(data, os.path.join(self.checkpoint_dir, save_path))
            log.info(f"\n Saved model with the highest evaluated average episode reward {self.current_best_reward:4.3f} to \n{save_path}\n ")
            self.is_best_so_far =False
    
    @torch.no_grad()
    def get_samples_logprobs(self, 
                             cond:dict, 
                             ret_device='cpu', 
                             save_chains=True, 
                             normalize_denoising_horizon=False, 
                             normalize_act_space_dimension=False, 
                             clip_intermediate_actions=True,
                             account_for_initial_stochasticity=True):
        # 返回说明：action_samples 仍为 numpy，因为 mujoco 引擎接收 np。
        if save_chains:
            action_samples, chains_venv, logprob_venv  = self.model.get_actions(cond, 
                                                                                eval_mode=self.eval_mode, 
                                                                                save_chains=save_chains, 
                                                                                normalize_denoising_horizon=normalize_denoising_horizon, 
                                                                                normalize_act_space_dimension=normalize_act_space_dimension, 
                                                                                clip_intermediate_actions=clip_intermediate_actions,
                                                                                account_for_initial_stochasticity=account_for_initial_stochasticity)        # [n_envs, horizon_steps, act_dim]
            return action_samples.cpu().numpy(), chains_venv.cpu().numpy() if ret_device=='cpu' else chains_venv, logprob_venv.cpu().numpy()  if ret_device=='cpu' else logprob_venv
        else:
            action_samples, logprob_venv  = self.model.get_actions(cond, 
                                                                   eval_mode=self.eval_mode, 
                                                                   save_chains=save_chains, 
                                                                   normalize_denoising_horizon=normalize_denoising_horizon, 
                                                                   normalize_act_space_dimension=normalize_act_space_dimension, 
                                                                   clip_intermediate_actions=clip_intermediate_actions,
                                                                   account_for_initial_stochasticity=account_for_initial_stochasticity)
            return action_samples.cpu().numpy(), logprob_venv.cpu().numpy()  if ret_device=='cpu' else logprob_venv
    
    def get_value(self, cond:dict, device='cpu'):
        # cond 包含位于 self.device 上的浮点 torch.tensor
        if device == 'cpu':
            value_venv = self.model.critic.forward(cond).cpu().numpy().flatten()
        else:
            value_venv = self.model.critic.forward(cond).squeeze().float().to(self.device)
        return value_venv
    
    # 重载
    def update_lr(self, val_metric=None):
        if self.target_kl and self.lr_schedule == 'adaptive_kl':   # 按每个 minibatch 的 KL 散度自适应调整学习率。
            return
        else: # 使用预定义学习率调度器。
            super().update_lr()
    
    def update_lr_adaptive_kl(self, approx_kl):
        min_actor_lr = 1e-5
        max_actor_lr = 5e-4
        min_critic_lr = 1e-5
        max_critic_lr = 1e-3
        tune='maintains'
        if approx_kl > self.target_kl * 2.0:
            self.actor_lr = max(min_actor_lr, self.actor_lr / 1.5)
            self.critic_lr = max(min_critic_lr, self.critic_lr / 1.5)
            tune = 'decreases'
        elif 0.0 < approx_kl and approx_kl < self.target_kl / 2.0:
            self.actor_lr = min(max_actor_lr, self.actor_lr * 1.5)
            self.critic_lr = min(max_critic_lr, self.critic_lr * 1.5)
            tune = 'increases'
        for actor_param_group, critic_param_group in zip(self.actor_optimizer.param_groups, self.critic_optimizer.param_groups):
            actor_param_group["lr"] = self.actor_lr
            critic_param_group["lr"] = self.critic_lr
        log.info(f"""adaptive kl {tune} lr: actor_lr={self.actor_optimizer.param_groups[0]["lr"]:.2e}, critic_lr={self.critic_optimizer.param_groups[0]["lr"]:.2e}""")
    
    def minibatch_generator(self):
        self.approx_kl = 0.0
        
        obs, chains, returns, oldvalues, advantages, oldlogprobs =  self.buffer.make_dataset()
        # 使用价值函数解释未来回报方差
        self.explained_var = self.buffer.get_explained_var(oldvalues, returns)
        
        self.total_steps = self.n_steps * self.n_envs
        for update_epoch in range(self.update_epochs):
            self.kl_change_too_much = False
            indices = torch.randperm(self.total_steps, device=self.device)
            if self.lr_schedule=='fixed' and self.kl_change_too_much:
                break
            for batch_id, start in enumerate(range(0, self.total_steps, self.batch_size)):
                end = start + self.batch_size
                inds_b = indices[start:end]
                minibatch = (
                    {"state": obs[inds_b]},
                    chains[inds_b],
                    returns[inds_b], 
                    oldvalues[inds_b],
                    advantages[inds_b],
                    oldlogprobs[inds_b] 
                )
                if self.lr_schedule=='fixed' and self.target_kl and self.approx_kl > self.target_kl: # 也可用 adaptive KL 代替提前停止。
                    self.kl_change_too_much = True
                    log.warning(f"KL change too much, approx_kl ={self.approx_kl} > {self.target_kl} = target_kl, stop optimization.")
                    break
                
                yield update_epoch, batch_id, minibatch    

    def minibatch_generator_repeat(self):
        self.approx_kl = 0.0
        
        obs, chains, returns, oldvalues, advantages, oldlogprobs =  self.buffer.make_dataset()
        # 使用价值函数解释未来回报方差
        self.explained_var = self.buffer.get_explained_var(oldvalues, returns)
        
        duplicate_multiplier = 10   # PPO diffusion 的 self.ft_denoising_steps。此设置用于严格对齐 PPODiffusion 的 batchsize。
        
        self.total_steps = self.n_steps * self.n_envs *  duplicate_multiplier
        
        for update_epoch in range(self.update_epochs):
            self.kl_change_too_much = False
            indices = torch.randperm(self.total_steps, device=self.device)
            if self.lr_schedule=='fixed' and self.kl_change_too_much:
                break
            for batch_id, start in enumerate(range(0, self.total_steps, self.batch_size)):
                end = start + self.batch_size
                inds_b = indices[start:end]
                batch_inds_b, denoising_inds_b = torch.unravel_index(
                    inds_b,
                    (self.n_steps * self.n_envs, duplicate_multiplier),
                )
                minibatch = (
                    {"state": obs[batch_inds_b]},
                    chains[batch_inds_b],
                    returns[batch_inds_b], 
                    oldvalues[batch_inds_b],
                    advantages[batch_inds_b],
                    oldlogprobs[batch_inds_b] 
                )
                if self.lr_schedule=='fixed' and self.target_kl and self.approx_kl > self.target_kl: # 也可用 adaptive KL 代替提前停止。
                    self.kl_change_too_much = True
                    log.warning(f"KL change too much, approx_kl ={self.approx_kl} > {self.target_kl} = target_kl, stop optimization.")
                    break
                
                yield update_epoch, batch_id, minibatch

    def agent_update(self, verbose=True):
        clipfracs_list = []
        noise_std_list = []
        for update_epoch, batch_id, minibatch in self.minibatch_generator() if not self.repeat_samples else self.minibatch_generator_repeat():

            # minibatch 梯度下降
            self.model: PPOFlow
            
            # print(f"minibatch 的形状: {minibatch[0]['state'].shape}. self.n_envs={self.n_envs}")
            pg_loss, entropy_loss, v_loss, bc_loss, \
            clipfrac, approx_kl, ratio, \
            oldlogprob_min, oldlogprob_max, oldlogprob_std, \
                newlogprob_min, newlogprob_max, newlogprob_std, \
                noise_std, Q_values= self.model.loss(*minibatch, 
                                                    use_bc_loss=self.use_bc_loss, 
                                                    bc_loss_type=self.bc_loss_type, normalize_denoising_horizon=self.normalize_denoising_horizon, 
                                                    normalize_act_space_dimension=self.normalize_act_space_dim,
                                                    verbose=verbose,
                                                    clip_intermediate_actions=self.clip_intermediate_actions,
                                                    account_for_initial_stochasticity=self.account_for_initial_stochasticity)
            self.approx_kl = approx_kl
            if verbose:
                log.info(f"update_epoch={update_epoch}/{self.update_epochs}, batch_id={batch_id}/{max(1, self.total_steps // self.batch_size)}, ratio={ratio:.3f}, clipfrac={clipfrac:.3f}, approx_kl={self.approx_kl:.2e}")
            
            if update_epoch ==0  and batch_id ==0 and np.abs(ratio-1.00)> self.initial_ratio_error_threshold:
                raise ValueError(f"ratio={ratio} not 1.00 when update_epoch ==0  and batch_id ==0, there must be some bugs in your code not related to hyperparameters !")
            
            if self.target_kl and self.lr_schedule == 'adaptive_kl':
                self.update_lr_adaptive_kl(self.approx_kl)
            
            loss = pg_loss + entropy_loss * self.ent_coef + v_loss * self.vf_coef + bc_loss * self.bc_coeff
            
            clipfracs_list += [clipfrac]
            noise_std_list += [noise_std]
            
            # 更新策略和 critic
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            loss.backward()
            
            # 调试损失项
            actor_norm = torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), max_norm=float('inf'))
            actor_old_norm = torch.nn.utils.clip_grad_norm_(self.model.actor_old.parameters(), max_norm=float('inf'))
            critic_norm = torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), max_norm=float('inf'))
            if verbose:
                log.info(f"before clipping: actor_norm={actor_norm:.2e}, critic_norm={critic_norm:.2e}, actor_old_norm={actor_old_norm:.2e}")
            
            # 始终高频更新 critic
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            
            # critic 预热后，为保证价值估计稳定，actor 低频但多次更新。
            if self.itr >= self.n_critic_warmup_itr:
                if (self.itr-self.n_critic_warmup_itr) % self.actor_update_freq ==0:
                    for _ in range(self.actor_update_epoch):
                        if self.max_grad_norm:
                            torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), self.max_grad_norm)
                        self.actor_optimizer.step()
        
        clip_fracs=np.mean(clipfracs_list)
        noise_stds=np.mean(noise_std_list)
        self.train_ret_dict = {
                "loss": loss,
                "pg loss": pg_loss,
                "value loss": v_loss,
                "entropy_loss": entropy_loss,
                "bc_loss": bc_loss,
                "approx kl": self.approx_kl,
                "ratio": ratio,
                "clipfrac": clip_fracs,
                "explained variance": self.explained_var,
                "old_logprob_min": oldlogprob_min,
                "old_logprob_max": oldlogprob_max,
                "old_logprob_std": oldlogprob_std,
                "new_logprob_min": newlogprob_min,
                "new_logprob_max": newlogprob_max,
                "new_logprob_std": newlogprob_std,
                "actor_norm": actor_norm,
                "critic_norm": critic_norm,
                "actor lr": self.actor_optimizer.param_groups[0]["lr"],
                "critic lr": self.critic_optimizer.param_groups[0]["lr"],
                "min_logprob_noise_std": self.model.min_logprob_denoising_std,
                "min_sampling_noise_std": self.model.min_sampling_denoising_std,
                "noise_std": noise_stds,
                "Q_values": Q_values
            }
    
    
