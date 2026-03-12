import numpy as np

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalVonNeumannGrid

from OpinionAgent import OpinionAgent

class OpinionWorld(Model):
    "Model class for the Opinion World model"

    EXTREME_TOLERANCE = 0.9

    def __init__(
            self,
            lattice_side_length = 21,
            initialisation_type = "uniform",
            neighbourhood_size = None,
            learning_rate = 0.4,
            confidence_threshold = 2,
            stubborn_proportion = 0.1,  # [新增] 顽固派选民的比例，默认 0.1 (10%)
            seed=None,
        ):
        if not (learning_rate > 0) or (learning_rate > 0.5):
            raise ValueError("Learning rate must be in the interval (0, 0.5]")
        
        super().__init__(seed=seed)

        # Model parameters
        self.num_agents = lattice_side_length**2
        self.stubborn_proportion = stubborn_proportion # [新增] 存为模型属性，方便后续调用或记录

        # Intialise environment, single capacity, toroidal boundaries
        self.grid = OrthogonalVonNeumannGrid(
            dimensions=(lattice_side_length, lattice_side_length),
            torus=True,
            random=self.random
            )

        # [新增] 初始化一个全为 False 的列表，用于标记每个人是否为顽固派
        is_stubborn_list = [False] * self.num_agents

        # set agent initial opinion distributions based on type
        match initialisation_type:
            case "uniform":
                opinions = self.rng.uniform(low=-1.0, high=1.0, size=self.num_agents)
            case "normal": # [新增] Q3(d) 的正态分布实现
                # 使用均值为0，标准差为0.3的正态分布
                opinions = self.rng.normal(loc=0.0, scale=0.3, size=self.num_agents)
                # 裁剪到 [-1.0, 1.0] 范围内，防止观点越界
                opinions = np.clip(opinions, -1.0, 1.0)
            case "stubborn": # [新增] 处理顽固派初始化的逻辑
                # 先让所有人按均匀分布生成观点
                opinions = self.rng.uniform(low=-1.0, high=1.0, size=self.num_agents)
                # 计算出顽固派的具体人数
                num_stubborn = int(self.num_agents * self.stubborn_proportion)
                # 随机抽取这些人的索引 (replace=False 保证不会重复抽到同一个人)
                stubborn_indices = self.rng.choice(self.num_agents, num_stubborn, replace=False)
                
                # 修改这些被抽中者的属性
                for idx in stubborn_indices:
                    opinions[idx] = -1.0          # 强制设定为强烈反对 (-1.0)
                    is_stubborn_list[idx] = True  # 将其标记为顽固派
            case _:
                raise ValueError(
                    f"unknown value of initialisation type: {initialisation_type}"
                )
        
        # if no neighbourhood size is given, assume agents are fully connected
        if neighbourhood_size is None:
            neighbourhood_size = lattice_side_length

        # Initialise agents
        OpinionAgent.create_agents(
            self,
            self.num_agents,
            self.grid.all_cells.cells,
            opinions,
            learning_rate,
            neighbourhood_size,
            confidence_threshold,
            is_stubborn_list # [新增] 将这个状态列表传入 Agent 创建函数
            )

        # Create data collectors
        self.datacollector = DataCollector(
            model_reporters={"opinions_mean": lambda m: np.mean([a.opinion for a in m.agents]),
                           "opinions_std": lambda m: np.std([a.opinion for a in m.agents]),
                           "proportion_of_extremists": lambda m: sum([1 for a in m.agents if m.EXTREME_TOLERANCE < abs(a.opinion)]) / m.num_agents
                           },
            agent_reporters={"opinion": lambda a: a.opinion
                             },
        )
        self.datacollector.collect(self)

    def step(self):
        """
        Run one step of the model.
        """
        # All the agents chat
        self.agents.shuffle_do("chat")

        # Collect data
        self.datacollector.collect(self)