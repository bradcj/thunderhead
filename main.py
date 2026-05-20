import json
import os

# import numpy as np

# village_dtype = np.dtype(
#     [
#         ("name", "U20"),
#         ("population", "i4"),
#         ("resources", [("food", "i4"), ("water", "i4"), ("materials", "i4")]),
#     ]
# )

# villages = np.array(
#     [
#         ("River Village", 50, (100, 200, 50)),
#         ("Forest Village", 100, (50, 50, 400)),
#     ],
#     dtype=village_dtype,
# )

villages = [
    {
        "name": "River Village",
        "population": 50,
        "resources": {"food": 100, "water": 200, "materials": 50},
        "happiness": None,
    },
    {
        "name": "Forest Village",
        "population": 100,
        "resources": {"food": 50, "water": 50, "materials": 400},
        "happiness": None,
    },
]

FOOD_WEIGHT = 0.4
WATER_WEIGHT = 0.4
MATERIALS_WEIGHT = 0.2


def calculate_happiness(village):
    # happiness should be a percentage of how well the village is doing based on its resources and population
    food, water, materials, population = (
        village["resources"].get("food", 0),
        village["resources"].get("water", 0),
        village["resources"].get("materials", 0),
        village["population"],
    )
    if population == 0:
        return 0.0

    food_score = min(food / (population * 2), 1.0) * FOOD_WEIGHT
    water_score = min(water / (population * 2), 1.0) * WATER_WEIGHT
    materials_score = min(materials / (population * 1), 1.0) * MATERIALS_WEIGHT
    total_score = food_score + water_score + materials_score
    return total_score


def run_simulation(villages):
    for village in villages:
        happiness = calculate_happiness(village)
        village["happiness"] = happiness
        print(f"{village['name']} Happiness: {happiness:.2f}")


def save_state(villages, filename="village_state.json"):
    with open(filename, "w") as f:
        json.dump(villages, f, indent=4)


def load_state(filename="village_state.json"):
    if not os.path.exists(filename):
        return villages
    with open(filename, "r") as f:
        return json.load(f)


def main():
    villages = load_state()
    run_simulation(villages)
    save_state(villages)


if __name__ == "__main__":
    main()
