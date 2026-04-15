"""
所有实验的启动入口。按需下载预训练数据、归一化统计信息和预训练检查点。
由 ReinFlow 作者修订，以支持断点续训并修复 kitchen 任务导入错误。
"""
# 自动清理 Python 缓存。
from util.clear_pycache import clean_pycache
from util.dirs import REINFLOW_DIR
clean_pycache(directory=REINFLOW_DIR)

# 提前注册 kitchen 任务，防止出现环境未找到错误。
import gym
import d4rl.gym_mujoco

import gc
gc.collect()

import os
import sys
import logging
import math
import hydra
from omegaconf import OmegaConf
import gdown
from download_url import (
    get_dataset_download_url,
    get_normalization_download_url,
    get_checkpoint_download_url,
)
# 允许在配置中通过 ${eval:''} 解析器执行任意 Python 代码。
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

# 抑制 d4rl 导入报错。
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

# 初始化日志记录器。
log = logging.getLogger(__name__)

# 对 stdout 和 stderr 都使用行缓冲。
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "cfg"),  # 可能会被 --config-path 参数覆盖。
)
def main(cfg: OmegaConf):
    # 立即解析配置，确保所有 ${now:} 解析器使用同一时间戳。
    OmegaConf.resolve(cfg)
    
    # ReinFlow 作者：从配置文件设置渲染后端。
    sim_device = cfg.get('sim_device', None)
    if sim_device is not None:
        # 若包含 cuda: 前缀，则提取其中的数字设备索引。
        if isinstance(sim_device, str) and sim_device.startswith('cuda:'):
            try:
                sim_device = int(sim_device.split('cuda:')[1])
            except ValueError:
                raise ValueError(f"Invalid sim_device format: {sim_device}. Expected 'cuda:<number>' or a numeric index.")
        elif isinstance(sim_device, (int, str)):
            try:
                sim_device = int(sim_device)  # 确保其为整数。
            except ValueError:
                raise ValueError(f"Invalid sim_device: {sim_device}. Must be a numeric GPU index.")
        else:
            raise ValueError(f"Invalid sim_device: {sim_device}. Must be a numeric GPU index.")

        os.environ['MUJOCO_GL'] = 'egl'
        os.environ['MUJOCO_sim_device_ID'] = str(sim_device)  # 环境变量值需要字符串类型。
        os.environ['sim_device_ID'] = str(sim_device)
        log.info(f"Set sim_device={sim_device} from cfg.")
    else:
        os.environ['MUJOCO_GL'] = 'osmesa'
        log.info("No EGL device specified in cfg, falling back to osmesa.")

    # 预训练场景：按需下载数据集。
    if "train_dataset_path" in cfg and not os.path.exists(cfg.train_dataset_path):
        download_url = get_dataset_download_url(cfg)
        download_target = os.path.dirname(cfg.train_dataset_path)
        log.info(f"Downloading dataset from {download_url} to {download_target}")
        gdown.download_folder(url=download_url, output=download_target)

    # 微调场景：按需下载归一化统计文件。
    if "normalization_path" in cfg and not os.path.exists(cfg.normalization_path):
        download_url = get_normalization_download_url(cfg)
        download_target = cfg.normalization_path
        dir_name = os.path.dirname(download_target)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        log.info(f"Downloading normalization statistics from {download_url} to {download_target}")
        gdown.download(url=download_url, output=download_target, fuzzy=True)

    # 微调场景：按需下载检查点。
    # ReinFlow 作者：若要从已有微调检查点继续训练，请将 base_policy_path 设为 null。
    if "base_policy_path" in cfg and cfg.base_policy_path and (not os.path.exists(cfg.base_policy_path)):
        download_url = get_checkpoint_download_url(cfg)
        if download_url is None:
            raise ValueError(f"Unknown checkpoint path {cfg.base_policy_path}. Did you specify the correct path to the policy you trained?")
        download_target = cfg.base_policy_path
        dir_name = os.path.dirname(download_target)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        log.info(f"Downloading checkpoint from {download_url} to {download_target}")
        gdown.download(url=download_url, output=download_target, fuzzy=True)

    # 处理 isaacgym 需在 torch 之前导入的依赖顺序问题。
    if "env" in cfg and "env_type" in cfg.env and cfg.env.env_type == "furniture":
        import furniture_bench
        # import torch
        # torch.cuda.empty_cache()
    
    # 运行智能体。
    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg)
    agent.run()


if __name__ == "__main__":
    main()
