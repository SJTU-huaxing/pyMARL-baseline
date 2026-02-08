import numpy as np
from typing import Optional

class DecayThenFlatSchedule(): #先衰减后平坦的调度器，常用于探索度的衰减
    #TODO ：探索现代的更好的衰减方式
    
    def __init__(self,
                 start: float,
                 finish: float,
                 time_length: float | int,
                 decay: str = "exp"):
        self.start = start #初始值
        self.finish = finish #结束值
        self.time_length = time_length #衰减时间长度
        self.delta = (self.start - self.finish) / self.time_length #每个时间步的衰减量
        self.decay = decay #衰减类型

        def eval(self, T: float | int) -> float: #根据时间计算当前值
            if self.decay == "linear": #线性衰减
                return max(self.finish, self.start - self.delta * T) 
            elif self.decay == "exp": #指数衰减
                return max(self.finish, self.finish + (self.start - self.finish) * np.exp(- T / self.time_length)) 
        pass