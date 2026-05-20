from enum import Enum
import json
import os
import numpy as np

import mesa
from mesa.discrete_space import CellAgent, OrthogonalMooreGrid

initial_state = {
    "villages": [
        {
            "name": "River Village",
            "population": 50,
            "resources": {"food": 100, "water": 200, "materials": 50},
            "location": {"x": 2, "y": 2},
            "happiness": None,
        },
        {
            "name": "Forest Village",
            "population": 100,
            "resources": {"food": 50, "water": 50, "materials": 400},
            "location": {"x": 3, "y": 3},
            "happiness": None,
        },
    ],
    "environment": [
        {
            "type": "river",
            "location": {"x": 2, "y": 2},
            "food": 100,
            "water": 400,
            "materials": 0,
        },
        {
            "type": "forest",
            "location": {"x": 3, "y": 3},
            "food": 50,
            "water": 0,
            "materials": 400,
        },
    ],
}


class EnvironmentType(Enum):
    RIVER = "river"
    FOREST = "forest"


class EnvironmentAgent(CellAgent):
    def __init__(self, model, cell, env_type: EnvironmentType, food, water, materials):
        super().__init__(model)
        self.cell = cell
        self.type = env_type
        self.food = food
        self.water = water
        self.materials = materials


class VillageAgent(CellAgent):
    FOOD_WEIGHT = 0.4
    WATER_WEIGHT = 0.4
    MATERIALS_WEIGHT = 0.2

    def __init__(self, model, cell, name, population, resources):
        super().__init__(model)
        self.cell = cell
        self.name = name
        self.population = population
        self.resources = resources
        self.happiness = self.calculate_happiness()

    def calculate_happiness(self):
        food, water, materials, population = (
            self.resources.get("food", 0),
            self.resources.get("water", 0),
            self.resources.get("materials", 0),
            self.population,
        )

        if population == 0:
            return 0.0

        food_score = min(food / (population * 2), 1.0) * self.FOOD_WEIGHT
        water_score = min(water / (population * 2), 1.0) * self.WATER_WEIGHT
        materials_score = min(materials / (population * 1), 1.0) * self.MATERIALS_WEIGHT
        total_score = food_score + water_score + materials_score
        return total_score


class WorldModel(mesa.Model):
    def __init__(self, width, height, villages, environment, seed=None):
        super().__init__(seed=seed)
        self.num_villages = len(villages)
        self.num_environment = len(environment)
        self.grid = OrthogonalMooreGrid(
            (width, height), torus=False, random=self.random
        )

        for env in environment:
            env_agent = EnvironmentAgent(
                self,
                self.grid[env["location"]["x"], env["location"]["y"]],
                EnvironmentType(env["type"]),
                env["food"],
                env["water"],
                env["materials"],
            )

        for village in villages:
            village_agent = VillageAgent(
                self,
                self.grid[village["location"]["x"], village["location"]["y"]],
                village["name"],
                village["population"],
                village["resources"],
            )

    def step(self):
        # calculate/update happiness for all villages
        village_agents = self.agents.select(agent_type=VillageAgent)
        for village in village_agents:
            village.happiness = village.calculate_happiness()
            print(
                f"{village.name} has population={village.population}, resources={village.resources}, happiness={village.happiness:.2f}"
            )
            neighbors = village.cell.get_neighborhood(include_center=False)
            for neighbor_cell in neighbors:
                for neighbor in neighbor_cell.agents:
                    if isinstance(neighbor, EnvironmentAgent):
                        print(
                            f"{village.name} is near a {neighbor.type.value} with resources: food={neighbor.food}, water={neighbor.water}, materials={neighbor.materials}"
                        )
                    elif isinstance(neighbor, VillageAgent):
                        print(
                            f"{village.name} is near {neighbor.name} with population={neighbor.population} and happiness={neighbor.happiness:.2f}"
                        )

    def to_json(self):
        return {
            "villages": [
                {
                    "name": agent.name,
                    "population": agent.population,
                    "resources": agent.resources,
                    "location": {
                        "x": agent.cell.coordinate[0],
                        "y": agent.cell.coordinate[1],
                    },
                    "happiness": agent.happiness,
                }
                for agent in self.agents.select(agent_type=VillageAgent)
            ],
            "environment": [
                {
                    "type": agent.type.value,
                    "location": {
                        "x": agent.cell.coordinate[0],
                        "y": agent.cell.coordinate[1],
                    },
                    "food": agent.food,
                    "water": agent.water,
                    "materials": agent.materials,
                }
                for agent in self.agents.select(agent_type=EnvironmentAgent)
            ],
        }


STATE_FILENAME = "state.json"


def save_state(state, filename=STATE_FILENAME):
    with open(filename, "w") as f:
        json.dump(state, f, indent=4)


def load_state(filename=STATE_FILENAME):
    if not os.path.exists(filename):
        return initial_state
    with open(filename, "r") as f:
        return json.load(f)


def main():
    state = load_state()

    model = WorldModel(5, 5, state["villages"], state["environment"])
    model.step()

    save_state(model.to_json())


if __name__ == "__main__":
    main()
