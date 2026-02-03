"""
Weights & Biases (wandb) 日志记录工具
支持从配置文件读取设置，自动管理训练日志
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List


class WandbLogger:
    """
    wandb 日志记录器
    自动处理配置、初始化和日志记录
    """
    
    def __init__(
        self,
        enabled: bool = True,
        project: str = "VLP-LSTM-LB",
        entity: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """
        初始化 wandb 日志记录器
        
        Args:
            enabled: 是否启用 wandb
            project: wandb 项目名称
            entity: wandb 用户名/组织名
            api_key: wandb API 密钥（None 则从环境变量读取）
            config: 要记录的配置字典
            tags: 实验标签
            notes: 实验备注
            name: 实验名称（None 则自动生成）
        """
        self.enabled = enabled
        self.run = None
        
        if not enabled:
            print("[WandbLogger] Disabled")
            return
        
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            print("[WandbLogger] Warning: wandb not installed. Run: pip3 install wandb")
            self.enabled = False
            return
        
        # 设置 API 密钥（如果提供）
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
        
        # 登录检查
        try:
            if not self.wandb.login():
                print("[WandbLogger] Warning: wandb login failed. Training will continue without logging.")
                self.enabled = False
                return
        except Exception as e:
            print(f"[WandbLogger] Warning: wandb login error: {e}")
            self.enabled = False
            return
        
        # 初始化 run
        try:
            self.run = self.wandb.init(
                project=project,
                entity=entity,
                config=config,
                tags=tags,
                notes=notes,
                name=name,
                reinit=True,  # 允许在 Jupyter 中重新初始化
            )
            print(f"[WandbLogger] Initialized: {self.run.url}")
        except Exception as e:
            print(f"[WandbLogger] Error initializing: {e}")
            self.enabled = False
    
    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        """记录指标"""
        if self.enabled and self.run:
            self.wandb.log(data, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """记录训练指标（带 step）"""
        self.log(metrics, step=step)
    
    def log_config(self, config: Dict[str, Any]):
        """更新配置"""
        if self.enabled and self.run:
            self.wandb.config.update(config)
    
    def log_artifact(self, artifact_path: str, artifact_type: str = "model", name: Optional[str] = None):
        """
        记录模型 artifact
        
        Args:
            artifact_path: 文件路径
            artifact_type: 类型（model, dataset, result 等）
            name: artifact 名称
        """
        if not self.enabled or not self.run:
            return
        
        try:
            artifact = self.wandb.Artifact(
                name=name or f"{artifact_type}-{self.run.id}",
                type=artifact_type
            )
            artifact.add_file(artifact_path)
            self.run.log_artifact(artifact)
            print(f"[WandbLogger] Artifact logged: {artifact_path}")
        except Exception as e:
            print(f"[WandbLogger] Error logging artifact: {e}")
    
    def log_figure(self, figure, name: str):
        """记录 matplotlib 图表"""
        if self.enabled and self.run:
            self.wandb.log({name: self.wandb.Image(figure)})
    
    def finish(self):
        """结束日志记录"""
        if self.enabled and self.run:
            self.run.finish()
            print("[WandbLogger] Finished")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    从 YAML 文件加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    import yaml
    
    if not Path(config_path).exists():
        print(f"[Config] {config_path} not found, using default config")
        return {
            "wandb": {
                "enabled": False,
                "project": "VLP-LSTM-LB",
                "entity": None,
            }
        }
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def create_logger_from_config(
    config_path: str = "config.yaml",
    model_name: str = "v2",
    extra_config: Optional[Dict[str, Any]] = None
) -> WandbLogger:
    """
    从配置文件创建 wandb 日志记录器
    
    Args:
        config_path: 配置文件路径
        model_name: 模型名称（用于实验名）
        extra_config: 额外的配置项
        
    Returns:
        WandbLogger 实例
    """
    config = load_config(config_path)
    wandb_config = config.get("wandb", {})
    
    # 合并配置
    merged_config = {
        "model": model_name,
        **(extra_config or {}),
    }
    
    # 添加 training 和 model 配置
    if "training" in config:
        merged_config["training"] = config["training"]
    if "model" in config:
        merged_config["model_config"] = config["model"]
    
    # 从环境变量读取 API 密钥（优先级高于配置文件）
    api_key = os.getenv("WANDB_API_KEY") or wandb_config.get("api_key")
    
    return WandbLogger(
        enabled=wandb_config.get("enabled", True),
        project=wandb_config.get("project", "VLP-LSTM-LB"),
        entity=wandb_config.get("entity"),
        api_key=api_key,
        config=merged_config,
        tags=wandb_config.get("tags", []),
        notes=wandb_config.get("notes"),
        name=f"{model_name}-{wandb.util.generate_id()[:8]}" if wandb_config.get("enabled") else None,
    )


# 简化导入
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
