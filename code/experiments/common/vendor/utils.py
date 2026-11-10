# coding=utf-8
"""共享的 PDE 元信息容器。

发布包只保留 ``model.py`` 实际 import 的 ``PDE``；上游同一模块里的
Dataset / GraphCreator / 计时器等重复实现已移除，数据集与图构建统一走
``utils_tubor.py``。
"""
from typing import Tuple


class PDE(object):
    def __init__(self,
                 name: str,
                 variables: dict,
                 temporal_domain: Tuple,
                 resolution_t: int,
                 spatial_domain: list,
                 resolution: list,
                 reduced_resolution_t: int = 1,
                 reduced_resolution: int = 1):
        super().__init__()
        self.name = name
        self.tmin = temporal_domain[0]
        self.tmax = temporal_domain[1]
        self.resolution_t = resolution_t // reduced_resolution_t
        self.spatial_domain = spatial_domain
        self.resolution = [(res - 1) // reduced_resolution + 1 for res in resolution]
        self.spatial_dim = len(spatial_domain)
        self.variables = variables

    def __repr__(self):
        return self.name
