import torch
from torch import nn
import copy
import torch.nn.functional as F
from torch import Tensor
import logging
log = logging.getLogger(__name__)
from collections import namedtuple
from typing import Tuple
from torch.distributions.normal import Normal
from model.flow.mlp_flow import FlowMLP, NoisyFlowMLP
Sample = namedtuple("Sample", "trajectories chains")

class PPOFlow(nn.Module):
    def __init__(self, 
                 device,
                 policy,
                 critic,
                 actor_policy_path,
                 act_dim,
                 horizon_steps,
                 act_min, 
                 act_max,
                 obs_dim,
                 cond_steps,
                 noise_scheduler_type,
                 inference_steps,
                 ft_denoising_steps,
                 randn_clip_value,
                 min_sampling_denoising_std,
                 min_logprob_denoising_std,
                 logprob_min,
                 logprob_max,
                 clip_ploss_coef,
                 clip_ploss_coef_base,
                 clip_ploss_coef_rate,
                 clip_vloss_coef,
                 denoised_clip_value,
                 max_logprob_denoising_std,
                 time_dim_explore,
                 learn_explore_time_embedding,
                 use_time_independent_noise,
                 noise_hidden_dims,
                 logprob_debug_sample,
                 logprob_debug_recalculate,
                 explore_net_activation_type
                 ):
        
        super().__init__()
        self.device = device
        self.inference_steps = inference_steps          # 推理阶段的步数。
        self.ft_denoising_steps = ft_denoising_steps    # 可调的微调去噪步数。
        self.action_dim = act_dim
        self.horizon_steps = horizon_steps
        self.act_dim_total = self.horizon_steps * self.action_dim
        self.act_min = act_min
        self.act_max = act_max
        
        self.obs_dim = obs_dim
        self.cond_steps = cond_steps
        
        self.noise_scheduler_type:str = noise_scheduler_type

        # 防止高斯采样出现极端值。偏离均值的范围限制在 `randn_clip_value` 倍标准差以内。
        self.randn_clip_value:float = randn_clip_value
        
        # 动作采样时去噪过程使用的最小标准差，有助于探索。
        self.min_sampling_denoising_std:float = min_sampling_denoising_std

        # 计算去噪 logprob 时使用的标准差上下限，用于提升稳定性。
        self.min_logprob_denoising_std:float = min_logprob_denoising_std
        self.max_logprob_denoising_std:float = max_logprob_denoising_std
        
        # 每个 batch 中 logprob 的上下界；裁剪到该范围内以避免策略坍塌。
        self.logprob_min:float= logprob_min
        self.logprob_max:float= logprob_max
        
        self.clip_ploss_coef:float = clip_ploss_coef
        self.clip_ploss_coef_base:float = clip_ploss_coef_base
        self.clip_ploss_coef_rate:float = clip_ploss_coef_rate
        self.clip_vloss_coef:float = clip_vloss_coef
        
        # 推理过程中裁剪中间动作。
        self.denoised_clip_value:float = denoised_clip_value
        self.logprob_debug_sample=logprob_debug_sample
        self.logprob_debug_recalculate=logprob_debug_recalculate
        
        # 噪声网络相关设置。
        self.learn_explore_time_embedding=learn_explore_time_embedding
        self.time_dim_explore=time_dim_explore
        self.use_time_independent_noise=use_time_independent_noise
        self.noise_hidden_dims=noise_hidden_dims
        self.explore_net_activation_type=explore_net_activation_type
        
        self.actor_old: FlowMLP = policy
        self.load_policy(actor_policy_path, use_ema=True)  # 早期 hopper/walker/halfcheetah 实验中此项为 False。
        for param in self.actor_old.parameters():
            param.requires_grad = False             # 不训练该副本，仅用于加载检查点。
        self.actor_old.to(self.device)
        
        policy_copy = copy.deepcopy(self.actor_old)
        for param in policy_copy.parameters():
            param.requires_grad = True
        
        self.init_actor_ft(policy_copy)
        logging.info("Cloned policy for fine-tuning")
        
        self.critic = critic
        self.critic = self.critic.to(self.device)
        
        self.report_network_params()
    
    def init_actor_ft(self, policy_copy):
        self.actor_ft = NoisyFlowMLP(policy=policy_copy,
                                    denoising_steps=self.inference_steps,
                                    learn_explore_noise_from = self.inference_steps - self.ft_denoising_steps,
                                    inital_noise_scheduler_type=self.noise_scheduler_type,
                                    min_logprob_denoising_std = self.min_logprob_denoising_std,
                                    max_logprob_denoising_std = self.max_logprob_denoising_std,
                                    learn_explore_time_embedding=self.learn_explore_time_embedding,
                                    time_dim_explore=self.time_dim_explore,
                                    use_time_independent_noise=self.use_time_independent_noise,
                                    device=self.device,
                                    noise_hidden_dims=self.noise_hidden_dims,
                                    activation_type=self.explore_net_activation_type
                                    )
    
    def check_gradient_flow(self):
        print(f"{next(self.actor_ft.policy.parameters()).requires_grad}") # True
        print(f"{next(self.actor_ft.mlp_logvar.parameters()).requires_grad}") # True
        print(f"{next(self.actor_ft.time_embedding_explore.parameters()).requires_grad}") # True
        print(f"{self.actor_ft.logvar_min.requires_grad}") # False
        print(f"{self.actor_ft.logvar_max.requires_grad}") # False
        
    def report_network_params(self):
        logging.info(
            f"Number of network parameters: Total: {sum(p.numel() for p in self.parameters())/1e6} M. Actor:{sum(p.numel() for p in self.actor_old.parameters())/1e6} M. Actor (finetune) : {sum(p.numel() for p in self.actor_ft.parameters())/1e6} M. Critic: {sum(p.numel() for p in self.critic.parameters())/1e6} M"
        )
    
    def load_policy(self, network_path, use_ema=False):
        log.info(f"loading policy from %s" % network_path)
        if network_path:
            print(f"network_path={network_path}, self.device={self.device}")
            model_data = torch.load(network_path, map_location=self.device, weights_only=True)
            actor_network_data = {k.replace("network.", ""): v for k, v in model_data["model"].items()}
            if use_ema:
                ema_actor_network_data = {k.replace("network.", ""): v for k, v in model_data["ema"].items()}
                self.actor_old.load_state_dict(ema_actor_network_data)
                logging.info("Loaded ema actor policy from %s", network_path)
            else:
                self.actor_old.load_state_dict(actor_network_data)
                logging.info("Loaded actor policy from %s", network_path)
            print(f"actor_network_data={actor_network_data.keys()}")
        else:
            logging.warning("No actor policy path provided. Not loading any actor policy. Start from randomly initialized policy.")
    
    @torch.no_grad()
    def sample_first_point(self, B:int)->Tuple[torch.Tensor, torch.Tensor]:
        '''
        B: batch 大小
        输出:
            xt: 形状为 `[batchsize, self.horizon_steps, self.action_dim]` 的张量
            log_prob: 形状为 `[batchsize]` 的张量
        '''
        dist = Normal(torch.zeros(B, self.horizon_steps* self.action_dim), 1.0)
        xt= dist.sample()
        log_prob = dist.log_prob(xt).sum(-1).to(self.device)                    # 可选择 mean() 或 sum()
        xt=xt.reshape(B, self.horizon_steps, self.action_dim).to(self.device)
        return xt, log_prob
    
    def get_logprobs(self, 
                     cond:dict, 
                     x_chain:Tensor, 
                     get_entropy =False, 
                     normalize_denoising_horizon=False, 
                     normalize_act_space_dimension=False,
                     clip_intermediate_actions=True,
                     verbose_entropy_stats=True,
                     debug=True,
                     account_for_initial_stochasticity=False,
                     get_chains_stds=True
                     ):
        '''
        输入:
            x_chain: 形状为 `[batchsize, self.inference_steps+1, self.horizon_steps, self.action_dim]` 的张量
           
        输出:
            log_prob: 形状为 `[batchsize]` 的张量
            entropy_rate_est: 形状为 `[batchsize]` 的张量
            chains_stds.mean(): 形状为 `[batchsize]` 的张量
            
        说明:
            p(x0|s)       = N(x0|0, 1)
            p(xt+1|xt, s) = N(xt+1 | xt + v(xt, s)1/K; sigma_t^2)
            
            log p(xK|s) = log p(x0) + \sum_{t=0}^{K-1} log p(xt+1|xt, s)
            H(X0:K)     = H(x0|s)     + \sum_{t=0}^{K-1} H(Xt+1|X_t, s)
            熵率 H(X) = H(X0:K)/(K+1)，当 K 趋于无穷时渐近收敛到每符号熵。
            我们将各维度和各时域上的动作视为在给定状态 s 与前一动作下条件独立。（开环执行）
        '''
        logprob = 0.0
        joint_entropy=0.0 
        entropy_rate_est=0.0
        logprob_steps = 0
        
        B = x_chain.shape[0]
        chains_prev = x_chain[:, :-1,:, :].flatten(-2,-1)                       # [batchsize, self.inference_steps, self.horizon_steps x self.action_dim]
        chains_next = x_chain[:, 1:, :, :].flatten(-2,-1)                       # [batchsize, self.inference_steps, self.horizon_steps x self.action_dim]
        chains_stds = torch.zeros_like(chains_prev, device=self.device)         # [batchsize, self.inference_steps, self.horizon_steps x self.action_dim]
        
        # 初始概率
        init_dist = Normal(torch.zeros(B, self.horizon_steps* self.action_dim, device=self.device), 1.0)
        logprob_init = init_dist.log_prob(x_chain[:,0].reshape(B,-1)).sum(-1)   # [batchsize]
        if get_entropy:
            entropy_init = init_dist.entropy().sum(-1)                          # [batchsize]
        if account_for_initial_stochasticity:
            logprob+=logprob_init
            if get_entropy:
                joint_entropy+=entropy_init
            logprob_steps+=1
        
        # 转移概率
        chains_vel  = torch.zeros_like(chains_prev, device=self.device)         # [batchsize, self.inference_steps, self.horizon_steps x self.action_dim]

        dt = 1.0/self.inference_steps
        steps = torch.linspace(0, 1-dt, self.inference_steps).repeat(B, 1).to(self.device)  # [batchsize, self.inference_steps]。linspace 默认包含左右端点，因此右端取 1-dt。
        for i in range(self.inference_steps):
            t       = steps[:,i]
            xt      = x_chain[:,i]                                              # [batchsize, self.horizon_steps , self.action_dim]
            vt, nt  =self.actor_ft.forward(xt, t, cond, True, i)                # [batchsize, self.horizon_steps, self.action_dim]
            chains_vel[:,i]  = vt.flatten(-2,-1)                                # [batchsize, self.horizon_steps x self.action_dim]
            chains_stds[:,i] = nt                                               # [batchsize, self.horizon_steps x self.action_dim]
            logprob_steps+=1
        chains_mean = (chains_prev + chains_vel * dt)                           # [batchsize, self.inference_steps, self.horizon_steps x self.action_dim]
        if clip_intermediate_actions:
            chains_mean = chains_mean.clamp(-self.denoised_clip_value, self.denoised_clip_value)
        
        # 转移分布
        chains_dist = Normal(chains_mean, chains_stds)
        
        # 转移项的对数概率与熵
        logprob_trans = chains_dist.log_prob(chains_next).sum(-1)               # [batchsize, self.inference_steps] 对 self.horizon_steps x self.action_dim 求和
        if get_entropy:
            entropy_trans = chains_dist.entropy().sum(-1)                       # [batchsize, self.inference_steps] 对所有维度求和
        
        # 整条马尔可夫链的对数概率
        logprob += logprob_trans.sum(-1)                          # [batchsize] 在推理步上累加（马尔可夫性质）
        if self.logprob_debug_recalculate: 
            log.info(f"logprob_init={logprob_init.mean().item()}, logprob_trans={logprob_trans.mean().item()}")
        # 整条马尔可夫链的熵率估计
        if get_entropy:
            joint_entropy +=entropy_trans.sum(-1)
        
        if get_entropy:
            entropy_rate_est = joint_entropy/logprob_steps
        if normalize_denoising_horizon:
            logprob = logprob / logprob_steps
            
        if normalize_act_space_dimension:
            logprob = logprob/self.act_dim_total
            if get_entropy:
                entropy_rate_est = entropy_rate_est/self.act_dim_total
        
        if verbose_entropy_stats and get_entropy:
            log.info(f"entropy_rate_est={entropy_rate_est.shape} Entropy Percentiles: 10%={entropy_rate_est.quantile(0.1):.2f}, 50%={entropy_rate_est.median():.2f}, 90%={entropy_rate_est.quantile(0.9):.2f}")
        
        if get_entropy:
            if get_chains_stds:
                return logprob, entropy_rate_est, chains_stds.mean()
            return logprob, entropy_rate_est, 
        else:
            if get_chains_stds:
                return logprob, chains_stds.mean()
            return logprob
    
    @torch.no_grad()
    def get_actions(self, 
                    cond:dict, 
                    eval_mode:bool, 
                    save_chains=False, 
                    normalize_denoising_horizon=False, 
                    normalize_act_space_dimension=False,
                    clip_intermediate_actions=True,
                    account_for_initial_stochasticity=True,
                    ret_logprob=True
                    ):
        '''
        从噪声初值出发，用 Euler 离散化 flow ODE，在每一步用高斯转移采样下一状态，得到整条动作轨迹。

        输入:
            cond:
                ``state``: ``[B, cond_steps, obs_dim]``，条件观测（B 为 batch，与并行环境数一致）。
            eval_mode:
                ``True``：每步取 **均值** ``dist.loc`` 作为下一状态（评估/确定性 rollout）。
                ``False``：从 ``Normal(mean, std)`` **采样** 下一状态，并做 randn 裁剪以保留探索。
            save_chains:
                是否缓存 ``x0..xK`` 全链，供 buffer 与 ``get_logprobs`` 重算旧策略概率。
            normalize_denoising_horizon / normalize_act_space_dimension:
                对累加后的标量 ``log_prob`` 分别除以（参与累加的步数）、``act_dim_total``，用于尺度稳定。
            clip_intermediate_actions:
                每步 drift 之后是否将 ``xt`` clamp 到 ``±denoised_clip_value``，抑制中间动作爆炸。
            account_for_initial_stochasticity:
                ``log_prob`` 是否包含 ``x0 ~ N(0,I)`` 的 ``log p(x0)``；与 ``ret_logprob``、loss 侧开关需一致。
            ret_logprob:
                若为 ``False``，只做前向采样，不返回 log 概率（步数统计等逻辑跳过）。

        输出（``ret_logprob=True`` 时）:
            ``xt``: ``[B, horizon_steps, action_dim]``，第 K 步之后的动作块（即策略输出）。
            ``x_chain``（当 ``save_chains=True``）: ``[B, inference_steps+1, horizon_steps, action_dim]``，
            第 0 帧为 ``x0``，第 ``i+1`` 帧为第 ``i`` 次转移后的 ``xt``。
            ``log_prob``: ``[B]``，每条轨迹一个标量；由初分布项（可选）与各步转移 ``log_prob`` 累加后再按需归一化。
        '''
        # 获取 batch 大小
        B = cond["state"].shape[0]
        # 将连续时间 [0,1] 均分为 inference_steps 段；dt 为每段步长（与 vt 相乘做 Euler 更新）。
        dt = (1/self.inference_steps)* torch.ones(B, self.horizon_steps, self.action_dim, device=self.device)
        # 一系列离散时间点[0, 1/inference_steps, 2/inference_steps, ..., 1-1/inference_steps]
        steps = torch.linspace(0, 1-1/self.inference_steps,self.inference_steps).repeat(B, 1).to(self.device)  # [B, inference_steps]


        if save_chains:
            # 缓存整条轨迹，供 buffer 与 model.get_logprobs 重算旧策略概率。
            x_chain=torch.zeros((B, self.inference_steps+1, self.horizon_steps, self.action_dim), device=self.device)
        if ret_logprob:  # 如果 ret_logprob 为 True，则返回 log_prob，否则只返回动作
            log_prob=0.0        # 用于累计整条生成链的 log probability
            log_prob_steps=0    # 记录一共加了多少项，用于后续归一化
            if self.logprob_debug_sample: 
                log_prob_list = []
        
        # x0 ~ N(0,I)，与 sample_first_point 一致；log_prob_init 为 sum 维上的 log p(x0)。
        xt, log_prob_init = self.sample_first_point(B)  # 采样初始点 x
        if ret_logprob and account_for_initial_stochasticity:   # 如果要把初始噪声概率算进去
            log_prob+=log_prob_init
            log_prob_steps+=1
            if self.logprob_debug_sample:
                log_prob_list.append(log_prob_init.mean().item())
        
        xt:torch.Tensor
        if save_chains:
            x_chain[:, 0] = xt
        
        # 主循环，逐步 flow 更新
        for i in range(self.inference_steps):   # 对每一步转移项累加 log probability
            t = steps[:,i]

            # 输入：当前状态 xt、当前时间 t、条件观测 cond
            # 输出：
            #     vt：flow velocity，表示应该朝哪个方向更新
            #     nt：当前步的噪声尺度
            vt, nt =self.actor_ft.forward(xt, t, cond, learn_exploration_noise=False, step=i)   # 用 actor_ft 模型预测当前状态的 drift 与噪声
            # Euler 法更新当前状态 xt
            xt += vt* dt
            # 抑制中间态幅度，避免探索过大
            if clip_intermediate_actions: # 抑制中间态幅度，避免探索过大
                xt = xt.clamp(-self.denoised_clip_value, self.denoised_clip_value)
            
            # 用网络给出的噪声标量 nt 构造各维独立高斯；训练时采样，评估时用均值路径。
            std = nt.unsqueeze(-1).reshape(xt.shape)
            std = torch.clamp(std, min=self.min_sampling_denoising_std)    # 下界保证最小随机性
            dist = Normal(xt, std)  # Normal类 代表正态分布模型，构造转移分布
            if not eval_mode:  # 如果不在评估模式下，则从转移分布中采样；否则用均值路径（确定性路径）
                xt = dist.sample().clamp_(dist.loc - self.randn_clip_value * dist.scale,  # 用 clip 裁剪，防止探索过大
                                          dist.loc + self.randn_clip_value * dist.scale).to(self.device)
            
            # 最后一段再投影到合法动作区间（与任务 act_min/act_max 一致）。
            if i == self.inference_steps-1:
                xt = xt.clamp_(self.act_min, self.act_max)        

            if ret_logprob:  # 如果 ret_logprob 为 True，则返回 log_prob，否则只返回动作
                # 转移项：当前步采样（或均值）在 dist 下的 log p，对 horizon×act 维求和得到每条轨迹一个标量。
                logprob_transition = dist.log_prob(xt).sum(dim=(-2,-1)).to(self.device)  # 对 horizon×act 维求和得到每条轨迹一个标量
                if self.logprob_debug_sample: 
                    log_prob_list.append(logprob_transition.mean().item())  # 记录每一步 log_prob_transition 的均值
                log_prob += logprob_transition  # 累加 log_prob_transition
                log_prob_steps+=1   # 记录一共加了多少项，用于后续归一化
            if save_chains:
                x_chain[:, i+1] = xt    # 保存每一步链状态
        
        if ret_logprob:
            # 归一化 log_prob
            if normalize_denoising_horizon:
                log_prob = log_prob/log_prob_steps  # 除以参与累加的步数，减弱推理步数 K 变化带来的量级漂移
            if normalize_act_space_dimension:
                log_prob = log_prob/self.act_dim_total  # 再除以动作总维数，便于不同 act_dim 间调参
            if self.logprob_debug_sample:
                transform_logprob=torch.log(1-torch.tanh(x_chain[-1])**2+1e-7).sum(dim=(-2,-1)).mean().item()
                print(f"log_prob_list={log_prob_list}, transform={transform_logprob}")
        
        if ret_logprob:
            if save_chains:
                return (xt, x_chain, log_prob)  
            return (xt, log_prob)
        else:
            if save_chains:
                return (xt, x_chain)
            return xt
      
    
    def loss(
        self,
        obs,
        chains,
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        use_bc_loss=False,
        bc_loss_type='W2',
        normalize_denoising_horizon=False,
        normalize_act_space_dimension=False,
        verbose=True,
        clip_intermediate_actions=True,
        account_for_initial_stochasticity=True
    ):
        """
        PPO 损失
        obs: 以 state/rgb 为键的字典；越新的观测在最后
            "state": (B, To, Do)
            "rgb": (B, To, C, H, W)
        chains: (B, K+1, Ta, Da)
        returns: (B, )
        values: (B,)
        advantages: (B,)
        oldlogprobs: (B,)
        use_bc_loss: 是否增加 BC 正则损失
        normalize_act_space_dimension: 是否在所有时域步与动作维度上归一化 logprob 与熵率
        reward_horizon: 进行梯度反传的动作时域，当前暂未启用
        这里 B = n_steps x n_envs
        """
        
        newlogprobs, entropy, noise_std = self.get_logprobs(obs, 
                                                            chains, 
                                                            get_entropy=True, 
                                                            normalize_denoising_horizon=normalize_denoising_horizon,
                                                            normalize_act_space_dimension=normalize_act_space_dimension, 
                                                            verbose_entropy_stats=verbose, 
                                                            clip_intermediate_actions=clip_intermediate_actions,
                                                            account_for_initial_stochasticity=account_for_initial_stochasticity)
        if verbose:
            log.info(f"oldlogprobs.min={oldlogprobs.min():5.3f}, max={oldlogprobs.max():5.3f}, std of oldlogprobs={oldlogprobs.std():5.3f}")
            log.info(f"newlogprobs.min={newlogprobs.min():5.3f}, max={newlogprobs.max():5.3f}, std of newlogprobs={newlogprobs.std():5.3f}")
        
        
        newlogprobs = newlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)
        oldlogprobs = oldlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)
        if verbose:
            if oldlogprobs.min() < self.logprob_min: log.info(f"WARNINIG: old logprobs too low, potential policy collapse detected, should encourage exploration.")
            if newlogprobs.min() < self.logprob_min: log.info(f"WARNINIG: new logprobs too low, potential policy collapse detected, should encourage exploration.")
            if newlogprobs.max() > self.logprob_max: log.info(f"WARNINIG: new logprobs too high")
            if oldlogprobs.max() > self.logprob_max: log.info(f"WARNINIG: old logprobs too high")
        # 经验上我们观察到：当 logprob 的最小值过小（如低于 -3）或 std 大于 0.5 时（通常同时出现），
        # 性能会下降。
        # 对 advantages 做 batch 内归一化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if verbose:
            with torch.no_grad():
                advantage_stats = {
                    "mean":f"{advantages.mean().item():2.3f}",
                    "std": f"{advantages.std().item():2.3f}",
                    "max": f"{advantages.max().item():2.3f}",
                    "min": f"{advantages.min().item():2.3f}"
                }
                log.info(f"Advantage stats: {advantage_stats}")
                corr = torch.corrcoef(torch.stack([advantages, returns]))[0,1].item()
                log.info(f"Advantage-Reward Correlation: {corr:.2f}")
        
        # 计算比率
        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()
        
        # 计算 KL 差异与裁剪比例
        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_ploss_coef).float().mean().item()

        # 策略损失
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_ploss_coef, 1 + self.clip_ploss_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # 价值函数损失
        newvalues = self.critic(obs).view(-1)
        v_loss = 0.5 * ((newvalues - returns) ** 2).mean()
        if self.clip_vloss_coef: # 一般不建议启用。
            v_clipped = torch.clamp(newvalues, oldvalues -self.clip_vloss_coef, oldvalues + self.clip_vloss_coef)
            v_loss = 0.5 *torch.max((newvalues - returns) ** 2, (v_clipped - returns) ** 2).mean()
        if verbose:
            with torch.no_grad():
                mse = F.mse_loss(newvalues, returns)
                log.info(f"Value/Reward alignment: MSE={mse.item():.3f}")
        
        # 熵损失
        entropy_loss = -entropy.mean()
        # 监控策略熵分布
        if verbose:
            with torch.no_grad():
                log.info(f"Entropy Percentiles: 10%={entropy.quantile(0.1):.2f}, 50%={entropy.median():.2f}, 90%={entropy.quantile(0.9):.2f}")
        
        # BC 损失
        bc_loss = 0.0
        if use_bc_loss:
            if bc_loss_type=='W2':
                # 通过动作监督加入 Wasserstein 散度损失
                z=torch.zeros((obs['state'].shape[0], self.horizon_steps, self.action_dim), device=self.device)
                a_ω = self.actor_old.sample_action(cond=obs, inference_steps=self.inference_steps, clip_intermediate_actions=True, act_range=[self.act_min, self.act_max],z=z)
                a_θ = self.actor_ft.policy.sample_action(cond=obs, inference_steps=self.inference_steps, clip_intermediate_actions=True, act_range=[self.act_min, self.act_max],z=z)
                bc_loss = F.mse_loss(a_ω.detach(), a_θ)
            else:
                raise NotImplementedError
        return (
            pg_loss,
            entropy_loss,
            v_loss,
            bc_loss,
            clipfrac,
            approx_kl.item(),
            ratio.mean().item(),
            oldlogprobs.min(),
            oldlogprobs.max(),
            oldlogprobs.std(),
            newlogprobs.min(),
            newlogprobs.max(),
            newlogprobs.std(),
            noise_std.item(),
            newvalues.mean().item(),# Q 函数
        )
