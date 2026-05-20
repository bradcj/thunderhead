from enum import Enum
import json
import os
import numpy as np
import seaborn as sns

import mesa
from mesa.discrete_space import CellAgent, OrthogonalMooreGrid

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


class EnvironmentType(Enum):
    RIVER = "river"
    FOREST = "forest"


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
            "resources": self.resources,
        }


class VillageAgent(WorldAgent):
    FOOD_WEIGHT = 0.4
    WATER_WEIGHT = 0.4
    MATERIALS_WEIGHT = 0.2

    def __init__(self, model, cell, name, population, resources):
        super().__init__(model, cell, resources)
        self.name = name
        self.population = population
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

    def update_happiness(self):
        self.happiness = self.calculate_happiness()

    def harvest_resources(self):
        env_agents = [
            agent for agent in self.cell.agents if isinstance(agent, EnvironmentAgent)
        ]
        for env_agent in env_agents:
            # Harvest a portion of the resources from the environment
            harvested_food = int(env_agent.resources["food"] * 0.1)
            harvested_water = int(env_agent.resources["water"] * 0.1)
            harvested_materials = int(env_agent.resources["materials"] * 0.1)

            env_agent.resources["food"] -= harvested_food
            env_agent.resources["water"] -= harvested_water
            env_agent.resources["materials"] -= harvested_materials

            self.resources["food"] += harvested_food
            self.resources["water"] += harvested_water
            self.resources["materials"] += harvested_materials
            print(
                f"{self.name} harvested {harvested_food} food, {harvested_water} water, and {harvested_materials} materials from a {env_agent.type.value}"
            )

    def step(self):
        self.harvest_resources()
        self.update_happiness()

    def to_json(self):
        return {
            "type": "village",
            "name": self.name,
            "population": self.population,
            "resources": self.resources,
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
                    agent["resources"],
                )
            elif agent["type"] in EnvironmentType._value2member_map_:
                EnvironmentAgent(
                    self,
                    self.grid[agent["location"]["x"], agent["location"]["y"]],
                    EnvironmentType(agent["type"]),
                    agent["resources"],
                )

    def step(self):
        # calculate/update happiness for all villages
        village_agents = self.agents.select(agent_type=VillageAgent)
        for village in village_agents:
            village.step()
            print(
                f"{village.name} has population={village.population}, resources={village.resources}, happiness={village.happiness:.2f}"
            )
            neighbors = village.cell.get_neighborhood(include_center=False)
            for neighbor_cell in neighbors:
                for neighbor in neighbor_cell.agents:
                    if isinstance(neighbor, EnvironmentAgent):
                        print(
                            f"{village.name} is near a {neighbor.type.value} with resources: {neighbor.resources}"
                        )
                    elif isinstance(neighbor, VillageAgent):
                        print(
                            f"{village.name} is near {neighbor.name} with population={neighbor.population} and happiness={neighbor.happiness:.2f}"
                        )

    def to_json(self):
        return {"agents": [agent.to_json() for agent in self.agents]}


STATE_FILENAME = "state.json"


def save_state(state, filename=STATE_FILENAME):
    with open(filename, "w") as f:
        json.dump(state, f, indent=4)


def load_state(filename=STATE_FILENAME):
    if not os.path.exists(filename):
        return INITIAL_STATE
    with open(filename, "r") as f:
        return json.load(f)


def main():
    # state = load_state()
    state = INITIAL_STATE
    print(f"Initial state: {state}")

    model = WorldModel(5, 5, state["agents"])
    for _ in range(5):
        model.step()
        print(f"State after step {_ + 1}: {model.to_json()}")

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
