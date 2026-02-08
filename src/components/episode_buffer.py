import torch 
import numpy as np
from types import SimpleNamespace as SN
from typing import Dict, List, Tuple, Optional, Any

class EpisodeBatch:
    def __init__(self,
                 scheme: Dict[str, Any],
                 groups: Dict[str, Any],
                 batch_size: int,
                 max_seq_length: int,
                 data: Optional[SN] = None,
                 preprocess: Optional[Dict[str, Tuple[str, List[Any]]]] = None, #预处理数据
                 device: str = "cpu"):
        self.scheme =scheme.copy()
        self.groups = groups
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.preprocess = {} if preprocess is None else preprocess
        self.device = device

        if data is not None:
            self.data = data
        else:
            self.data = SN()
            self.data.transition_data = {} #每个时间步的数据
            self.data.episode_data = {} #每个episode的数据
            self._setup_data(self.scheme, self.groups, self.batch_size, self.max_seq_length, self.preprocess) 

    def _setup_data(self, scheme: Dict[str, Any],
                    groups: Dict[str, Any], 
                    batch_size: int, 
                    max_seq_length: int, 
                    preprocess: Optional[Dict[str, Tuple[str, List[Any]]]] = None) -> None:
        if preprocess is not None: 
            for k in preprocess:
                assert k in scheme
                new_k = preprocess[k][0]
                transforms = preprocess[k][1]

                vshape = self.scheme[k]['vshape']
                dtype = self.scheme[k]['dtype']
                for transform in transforms:
                    vshape, dtype = transform.infer_output_info(vshape, dtype)

                self.scheme[new_k] = {
                    'vshape': vshape, 
                    'dtype': dtype
                }
                if 'group' in self.scheme[k]:
                    self.scheme[new_k]['group'] = self.scheme[k]['group']
                if  'episode_const' in self.scheme[k]:
                    self.scheme[new_k]['episode_const'] = self.scheme[k]['episode_const']

        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update({
            "filled":{"vshape":(1,), "dtype": torch.long},
        })

        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, f"Scheme must define vshape for {field_key}"
            vshape = field_info["vshape"]
            episode_const = field_info.get("episode_const", False)
            group = field_info.get("group", None)
            dtype = field_info.get("dtype", torch.float32)

            if isinstance(vshape, int):
                vshape = (vshape,)

            if group:
                assert group in groups, f"Group {group} must have its number of members defined in _groups_"
                shape = (groups[group], *vshape)
            else:
                shape = vshape
            
            if episode_const:
                self.data.episode_data[field_key] = torch.zeros((batch_size, *shape), dtype=dtype, device=self.device)
            else:
                self.data.transition_data[field_key] = torch.zeros((batch_size, max_seq_length, *shape), dtype=dtype, device=self.device)

    def extend(self, scheme: Dict[str, Any], groups: Optional[Dict[str, Any]] = None) -> None:
        self._setup_data(scheme, self.groups if groups is None else groups, self.batch_size, self.max_seq_length)
    
    def to(self, device: str) -> None: 
        for k, v in self.data.transition_data.items():
            self.data.transition_data[k] = v.to(device)
        for k, v in self.data.episode_data.items():
            self.data.episode_data[k] = v.to(device)
    


