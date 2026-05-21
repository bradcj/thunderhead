from __future__ import annotations

from enum import Enum
import json
import math
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
    WATER_REPLENISH_RATE = 100
    # using simple rate for food/materials for now, can add more complex logic later (e.g. overconsumption reduces replenishment)
    FOOD_REPLENISH_RATE = 50
    MATERIALS_REPLENISH_RATE = 50

    def __init__(
        self,
        model,
        cell,
        env_type: EnvironmentType,
        resources,
    ):
        super().__init__(model, cell, resources)
        self.type = env_type

    def replenish_resources(self):
        self.resources[ResourceType.WATER] += self.WATER_REPLENISH_RATE
        self.resources[ResourceType.FOOD] += self.FOOD_REPLENISH_RATE
        self.resources[ResourceType.MATERIALS] += self.MATERIALS_REPLENISH_RATE
        logger.info(f"{self.type.value} replenished resources to {self.resources}")

    def to_json(self):
        return {
            "type": self.type.value,
            "location": {"x": self.cell.coordinate[0], "y": self.cell.coordinate[1]},
            "resources": {
                resource_type.value: amount
                for resource_type, amount in self.resources.items()
            },
        }


class RiverEnvironmentAgent(EnvironmentAgent):
    WATER_REPLENISH_RATE = 100
    FOOD_REPLENISH_MULTIPLIER = 1.5
    MATERIALS_REPLENISH_RATE = 0

    def __init__(self, model, cell, resources):
        super().__init__(
            model,
            cell,
            EnvironmentType.RIVER,
            resources,
        )


class ForestEnvironmentAgent(EnvironmentAgent):
    WATER_REPLENISH_RATE = 0
    FOOD_REPLENISH_MULTIPLIER = 1.5
    MATERIALS_REPLENISH_RATE = 50

    def __init__(self, model, cell, resources):
        super().__init__(
            model,
            cell,
            EnvironmentType.FOREST,
            resources,
        )


class RationLevel(Enum):
    ABUNDANT = "abundant"
    NORMAL = "normal"
    STARVATION = "starvation"


RATION_LEVEL_MULTIPLIERS = {
    RationLevel.ABUNDANT: 2.0,
    RationLevel.NORMAL: 1.0,
    RationLevel.STARVATION: 0.5,
}

CONSUMPTION_RATE_PER_PERSON = {
    ResourceType.FOOD: 2,
    ResourceType.WATER: 2,
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
        self.ration_level = RationLevel.NORMAL
        self.available_workers = population

    def replenish_workers(self):
        """Replenish available workers based on population"""
        self.available_workers = self.population

    def calculate_happiness(self):
        food, water, materials = (
            self.resources[ResourceType.FOOD],
            self.resources[ResourceType.WATER],
            self.resources[ResourceType.MATERIALS],
        )

        if self.population == 0:
            return 0.0

        food_score = min(food / (self.population * 2), 1.0) * self.FOOD_WEIGHT
        water_score = min(water / (self.population * 2), 1.0) * self.WATER_WEIGHT
        materials_score = (
            min(materials / (self.population * 1), 1.0) * self.MATERIALS_WEIGHT
        )
        total_score = food_score + water_score + materials_score
        return total_score

    def update_happiness(self):
        self.happiness = self.calculate_happiness()

    def set_ration_level(self, level: RationLevel):
        self.ration_level = level
        logger.info(f"{self.name} set ration level to {self.ration_level.value}")

    def consume_food_and_water(self):
        """Consume food and water based on population and ration level. If there is a shortage, reduce population accordingly."""
        for resource_type in [ResourceType.WATER, ResourceType.FOOD]:
            amount_needed = (
                self.population
                * CONSUMPTION_RATE_PER_PERSON[resource_type]
                * RATION_LEVEL_MULTIPLIERS[self.ration_level]
            )

            amount_consumed = min(self.resources[resource_type], amount_needed)
            if self.resources[resource_type] < amount_needed:
                shortage = amount_needed - self.resources[resource_type]
                people_affected = math.ceil(
                    shortage / CONSUMPTION_RATE_PER_PERSON[resource_type]
                )
                self.population = max(self.population - people_affected, 0)
                self.resources[resource_type] = 0
                logger.info(
                    f"{self.name} does not have enough {resource_type}. {people_affected} people have been affected. Remaining population: {self.population}"
                )
            else:
                self.resources[resource_type] -= amount_needed

            logger.info(
                f"{self.name} consumed {amount_consumed} {resource_type}. Remaining {resource_type}: {self.resources[resource_type]}"
            )

    def assign_workers_to_harvest(
        self,
        env_agent: EnvironmentAgent,
        resource_type: ResourceType,
        workers: int,
    ) -> int:
        """Assign workers to harvest a specific resource from an environment agent. Returns the number of workers actually assigned."""
        if self.available_workers <= 0:
            logger.warning(
                f"{self.name} has no available workers to harvest {resource_type} from {env_agent.type.value}."
            )
            return 0

        if self.available_workers < workers:
            logger.warning(
                f"{self.name} does not have enough available workers to harvest {resource_type} from {env_agent.type.value}. Using {self.available_workers} workers instead of {workers}."
            )
            workers = self.available_workers

        if env_agent.resources[resource_type] <= 0:
            logger.warning(
                f"{env_agent.type.value} does not have any {resource_type} left to harvest for {self.name}"
            )
            return 0

        harvest_amount = workers * self.WORKER_YIELD
        if harvest_amount > env_agent.resources[resource_type]:
            logger.warning(
                f"{env_agent.type.value} does not have enough {resource_type} to harvest {harvest_amount} for {self.name}. Using remaining {env_agent.resources[resource_type]} instead."
            )
            harvest_amount = env_agent.resources[resource_type]
            workers = math.ceil(harvest_amount / self.WORKER_YIELD)

        if harvest_amount <= 0:
            logger.warning(
                f"{env_agent.type.value} does not have any {resource_type} left to harvest for {self.name}"
            )
            return 0

        self.available_workers -= workers
        env_agent.resources[resource_type] -= harvest_amount
        self.resources[resource_type] += harvest_amount
        logger.info(
            f"{self.name} harvested {harvest_amount} {resource_type} from {env_agent.type.value} using {workers} workers. Remaining available workers: {self.available_workers}"
        )
        return workers

    def harvest_resources(self):
        """Harvest resources from all environment agents on the same cell."""
        env_agents = [
            agent for agent in self.cell.agents if isinstance(agent, EnvironmentAgent)
        ]
        for env_agent in env_agents:
            self.assign_workers_to_harvest(
                env_agent=env_agent,
                resource_type=ResourceType.WATER,
                workers=self.available_workers,
            )
            self.assign_workers_to_harvest(
                env_agent=env_agent,
                resource_type=ResourceType.FOOD,
                workers=self.available_workers,
            )

    def transfer_resource(
        self, target_village: VillageAgent, resource_type: ResourceType, amount: int
    ):
        if self.resources[resource_type] < amount:
            logger.warning(
                f"{self.name} does not have enough {resource_type} to transfer {amount} to {target_village.name}. Transferring remaining {self.resources[resource_type]} instead."
            )
            amount = self.resources[resource_type]

        if amount <= 0:
            logger.warning(
                f"{self.name} does not have any {resource_type} left to transfer to {target_village.name}"
            )
            return

        self.resources[resource_type] -= amount
        target_village.resources[resource_type] += amount
        logger.info(
            f"{self.name} transferred {amount} {resource_type} to {target_village.name}. Remaining {resource_type}: {self.resources[resource_type]}"
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
            elif agent["type"] == EnvironmentType.RIVER.value:
                RiverEnvironmentAgent(
                    self,
                    self.grid[agent["location"]["x"], agent["location"]["y"]],
                    {
                        ResourceType[key.upper()]: value
                        for key, value in agent["resources"].items()
                    },
                )
            elif agent["type"] == EnvironmentType.FOREST.value:
                ForestEnvironmentAgent(
                    self,
                    self.grid[agent["location"]["x"], agent["location"]["y"]],
                    {
                        ResourceType[key.upper()]: value
                        for key, value in agent["resources"].items()
                    },
                )
            else:
                raise ValueError(f"Unknown agent type: {agent['type']}")

    def step(self):
        env_agents = self.agents.select(agent_type=EnvironmentAgent)
        env_agents.do("replenish_resources")

        village_agents = self.agents.select(agent_type=VillageAgent)
        village_agents.do("replenish_workers")
        village_agents.shuffle_do("harvest_resources")
        village_agents.shuffle_do("consume_food_and_water")
        village_agents.do("update_happiness")

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
