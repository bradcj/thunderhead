from enum import Enum
import json
import os
import numpy as np
import seaborn as sns

import mesa
from mesa.discrete_space import CellAgent, OrthogonalMooreGrid

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("engine.main")

INITIAL_STATE = {
    "agents": [
        {
            "type": "village",
            "name": "River Village",
            "population": 50,
            "resources": {"food": 100, "water": 200, "materials": 50},
            "location": {"x": 1, "y": 1},
        },
        {
            "type": "village",
            "name": "Forest Village",
            "population": 100,
            "resources": {"food": 50, "water": 50, "materials": 400},
            "location": {"x": 3, "y": 3},
        },
        {
            "type": "river",
            "location": {"x": 1, "y": 1},
            "resources": {"food": 100, "water": 400, "materials": 0},
        },
        {
            "type": "forest",
            "location": {"x": 3, "y": 3},
            "resources": {"food": 50, "water": 0, "materials": 400},
        },
    ],
}


class ResourceType(Enum):
    FOOD = "food"
    WATER = "water"
    MATERIALS = "materials"

    def __str__(self):
        return self.value


class EnvironmentType(Enum):
    RIVER = "river"
    FOREST = "forest"

    def __str__(self):
        return self.value


class WorldAgent(CellAgent):
    def __init__(self, model, cell, resources):
        super().__init__(model)
        self.cell = cell
        self.resources = resources


class EnvironmentAgent(WorldAgent):
    def __init__(self, model, cell, env_type: EnvironmentType, resources):
        super().__init__(model, cell, resources)
        self.type = env_type

    def to_json(self):
        return {
            "type": self.type.value,
            "location": {"x": self.cell.coordinate[0], "y": self.cell.coordinate[1]},
            "resources": {
                resource_type.value: amount
                for resource_type, amount in self.resources.items()
            },
        }


class VillageAgent(WorldAgent):
    FOOD_WEIGHT = 0.4
    WATER_WEIGHT = 0.4
    MATERIALS_WEIGHT = 0.2

    WORKER_YIELD = 10  # 1 worker can harvest 10 units of resources per step

    def __init__(
        self, model, cell, name, population, resources: dict[ResourceType, int]
    ):
        super().__init__(model, cell, resources)
        self.name = name
        self.population = population
        self.happiness = self.calculate_happiness()

    def calculate_happiness(self):
        food, water, materials, population = (
            self.resources.get(ResourceType.FOOD, 0),
            self.resources.get(ResourceType.WATER, 0),
            self.resources.get(ResourceType.MATERIALS, 0),
            self.population,
        )

        if population == 0:
            return 0.0

        food_score = min(food / (population * 2), 1.0) * self.FOOD_WEIGHT
        water_score = min(water / (population * 2), 1.0) * self.WATER_WEIGHT
        materials_score = min(materials / (population * 1), 1.0) * self.MATERIALS_WEIGHT
        total_score = food_score + water_score + materials_score
        return total_score

    def update_happiness(self):
        self.happiness = self.calculate_happiness()

    def harvest_resource(
        self, env_agent: EnvironmentAgent, resource_type: ResourceType, amount: int
    ) -> int:
        """
        Attempt to harvest an amount and type of resources from a specific environment agent.
        Returns the actual amount harvested (which may be less than requested if not enough resources or available population).
        """
        workers = amount // self.WORKER_YIELD
        if self.available_workers < workers:
            logger.warning(
                f"{self.name} does not have enough available workers to harvest {amount} {resource_type} from {env_agent.type.value}. Using {self.available_workers} workers instead of {workers}."
            )

        if amount > env_agent.resources[resource_type]:
            logger.warning(
                f"{env_agent.type.value} does not have enough {resource_type} to harvest {amount} for {self.name}. Harvesting remaining {env_agent.resources[resource_type]} instead."
            )

        amount = min(amount, env_agent.resources.get(resource_type, 0))
        if amount <= 0:
            logger.warning(
                f"{env_agent.type.value} does not have any {resource_type} left to harvest for {self.name}"
            )
            return 0

        workers = min(workers, self.available_workers, amount // self.WORKER_YIELD)

        self.available_workers -= workers
        env_agent.resources[resource_type] -= amount
        self.resources[resource_type] += amount
        logger.info(
            f"{self.name} harvested {amount} {resource_type} from {env_agent.type.value} using {workers} workers. Remaining available workers: {self.available_workers}"
        )
        return amount

    def harvest_resources(self):
        env_agents = [
            agent for agent in self.cell.agents if isinstance(agent, EnvironmentAgent)
        ]
        for env_agent in env_agents:
            # # Harvest a portion of the resources from the environment
            # harvested_food = int(env_agent.resources["food"] * 0.1)
            # harvested_water = int(env_agent.resources["water"] * 0.1)
            # harvested_materials = int(env_agent.resources["materials"] * 0.1)

            # env_agent.resources["food"] -= harvested_food
            # env_agent.resources["water"] -= harvested_water
            # env_agent.resources["materials"] -= harvested_materials

            # self.resources["food"] += harvested_food
            # self.resources["water"] += harvested_water
            # self.resources["materials"] += harvested_materials
            # logger.info(
            #     f"{self.name} harvested {harvested_food} food, {harvested_water} water, and {harvested_materials} materials from a {env_agent.type.value}"
            # )
            env_agents = [
                agent
                for agent in self.cell.agents
                if isinstance(agent, EnvironmentAgent)
            ]
            for env_agent in env_agents:
                self.harvest_resource(
                    env_agent=env_agent,
                    resource_type=ResourceType.FOOD,
                    amount=self.available_workers * self.WORKER_YIELD,
                )

    def step(self):
        # assume all population is available to work
        self.available_workers = self.population
        self.harvest_resources()
        self.update_happiness()
        logger.info(
            f"{self.name} has population={self.population}, resources={self.resources}, happiness={self.happiness:.2f}"
        )

    def to_json(self):
        return {
            "type": "village",
            "name": self.name,
            "population": self.population,
            "resources": {
                resource_type.value: amount
                for resource_type, amount in self.resources.items()
            },
            "location": {"x": self.cell.coordinate[0], "y": self.cell.coordinate[1]},
            "happiness": self.happiness,
        }


class WorldModel(mesa.Model):
    def __init__(self, width, height, agents, seed=None):
        super().__init__(seed=seed)
        self.num_agents = len(agents)
        self.grid = OrthogonalMooreGrid(
            (width, height), torus=False, random=self.random
        )
        for agent in agents:
            if agent["type"] == "village":
                VillageAgent(
                    self,
                    self.grid[agent["location"]["x"], agent["location"]["y"]],
                    agent["name"],
                    agent["population"],
                    {
                        ResourceType[key.upper()]: value
                        for key, value in agent["resources"].items()
                    },
                )
            elif agent["type"] in EnvironmentType._value2member_map_:
                EnvironmentAgent(
                    self,
                    self.grid[agent["location"]["x"], agent["location"]["y"]],
                    EnvironmentType(agent["type"]),
                    {
                        ResourceType[key.upper()]: value
                        for key, value in agent["resources"].items()
                    },
                )

    def step(self):
        # calculate/update happiness for all villages
        village_agents = self.agents.select(agent_type=VillageAgent)
        village_agents.shuffle_do("step")

    def to_json(self):
        return {"agents": [agent.to_json() for agent in self.agents]}


STATE_FILENAME = "state.json"


def save_state(state, filename=STATE_FILENAME):
    with open(filename, "w") as f:
        logger.info(f"Saving state to {filename}: {state}")
        json.dump(state, f, indent=4)


def load_state(filename=STATE_FILENAME):
    if not os.path.exists(filename):
        return INITIAL_STATE
    with open(filename, "r") as f:
        return json.load(f)


def main():
    # state = load_state()
    state = INITIAL_STATE
    logger.info(f"Initial state: {state}")

    model = WorldModel(5, 5, state["agents"])
    for _ in range(5):
        model.step()
        logger.info(f"State after step {_ + 1}: {model.to_json()}")

    agent_counts = np.zeros((model.grid.width, model.grid.height))

    for cell in model.grid.all_cells:
        agent_counts[cell.coordinate] = len(cell.agents)
    g = sns.heatmap(agent_counts, cmap="viridis", annot=True, cbar=False, square=True)
    g.figure.set_size_inches(5, 5)
    g.set(title="Number of agents on each cell of the grid")
    g.figure.savefig("agent_counts.png")

    save_state(model.to_json())


if __name__ == "__main__":
    main()
