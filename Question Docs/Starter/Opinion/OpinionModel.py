import numpy as np

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalVonNeumannGrid

from .OpinionAgent import OpinionAgent

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
            seed=None,
        ):
        if not (learning_rate > 0) or (learning_rate > 0.5):
            raise ValueError("Learning rate must be in the interval (0, 0.5]")
        
        super().__init__(seed=seed)

        # Model parameters
        self.num_agents = lattice_side_length**2

        # Intialise environment, single capacity, toroidal boundaries
        self.grid = OrthogonalVonNeumannGrid(
            dimensions=(lattice_side_length, lattice_side_length),
            torus=True,
            random=self.random
            )

        # set agent initial opinion distributions based on type
        match initialisation_type:
            case "uniform":
                opinions = self.rng.uniform(low=-1.0, high=1.0, size=self.num_agents)
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
            confidence_threshold
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
