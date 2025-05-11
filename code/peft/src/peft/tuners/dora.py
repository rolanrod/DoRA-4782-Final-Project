import importlib
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import PeftConfig, PeftType, transpose


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class DoraConfig(PeftConfig):
    """Configuration class for DoraModel."""
    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None)
    lora_alpha: int = field(default=8)
    lora_dropout: float = field(default=0.0)
    merge_weights: bool = field(default=False)
    bias: str = field(default="none")

    def __post_init__(self):
        self.peft_type = PeftType.DORA


class LoraLayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else lambda x: x
        self.merged = False
        self.merge_weights = merge_weights


class Linear(nn.Linear, LoraLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.w_decomp= nn.Linear(1, out_features, bias=False)
        
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.scaling = self.lora_alpha / self.r

        self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
            nn.init.zeros_(self.lora_B.weight)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
       
        self.lora_A.train(mode)
        self.lora_B.train(mode)
        self.w_decomp.train(mode)

        if not mode and self.merge_weights and not self.merged:
            if self.r > 0:
                # Compute the adapted weights
                delta_weight = self.lora_B.weight @ self.lora_A.weight
                new_weight = self.weight + delta_weight * self.scaling

                # Normalize using the decomposition weights
                norm = torch.linalg.norm(new_weight, dim=1, keepdim=True)
                normalized_weight = (self.w_decomp.weight / norm) * new_weight

                # Update the weight tensor
                self.weight.data.copy_(normalized_weight.detach())
            self.merged = True

    def eval(self):
        nn.Linear.eval(self)
        self.lora_A.eval()
        self.lora_B.eval()
        self.w_decomp.eval()

    def forward(self, x: torch.Tensor):
        previous_dtype = self.weight.dtype
        
        if self.r > 0 and not self.merged:
            new_weight_v = self.weight + (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            
            norm_scale = self.w_decomp.weight.view(-1) / (torch.linalg.norm(new_weight_v, dim=1)).detach()
           
            org_result = F.linear(x, self.weight)
            dropout_x = self.lora_dropout(x)
            result = org_result + (norm_scale-1) * F.linear(dropout_x, self.weight)
            
            if self.bias is not None:
                result += self.bias.view(1, -1).expand_as(result)
                
            result += (norm_scale * self.lora_B(self.lora_A(dropout_x.to(self.lora_A.weight.dtype)))) * self.scaling
        else:
            result = F.linear(x, self.weight, bias=self.bias)

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)

        return result


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for name, param in model.named_parameters():
        if "lora_" not in name and "w_decomp" not in name:
            param.requires_grad = False
        else:
            print(f"{name} is trainable")
            
    if bias == "none":
        return
    elif bias == "all":
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True


class DoraModel(torch.nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_only_lora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        import re
    
        is_target_modules_in_base_model = False
        kwargs = {
            "r": self.peft_config.r,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "merge_weights": self.peft_config.merge_weights or self.peft_config.inference_mode,
        }
        
        for key, _ in self.model.named_modules():
            # Check if the module is a target module
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules or [])

            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                    
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None
                
                # Create appropriate module based on target type
                if isinstance(target, torch.nn.Linear):
                    new_module = Linear(
                        target.in_features, 
                        target.out_features, 
                        bias=bias, 
                        **kwargs
                    )
                    self._replace_module(parent, target_name, new_module, target)
                
        if not is_target_modules_in_base_model:
            raise ValueError(f"Target modules not found in the base model.")

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight

        # Initialize magnitude component
        with torch.no_grad():
            magnitude = torch.linalg.norm(new_module.weight.detach(), dim=1).unsqueeze(1).detach()
            new_module.w_decomp.weight.copy_(magnitude)
        
        if old_module.bias is not None:
            new_module.bias = old_module.bias
            
        # Dispatch modules to device
        for name, module in new_module.named_modules():
            if "lora_" in name or "w_decomp" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config