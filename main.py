villages = {
    "river_village": {
        "name": "River Village",
        "population": 50,
        "resources": {
            "food": 100,
            "water": 200,
            "materials": 50,
        },
    },
    "forest_village": {
        "name": "Forest Village",
        "population": 100,
        "resources": {
            "food": 50,
            "water": 50,
            "materials": 400,
        },
    },
}

FOOD_WEIGHT = 0.4
WATER_WEIGHT = 0.4
MATERIALS_WEIGHT = 0.2


def calculate_happiness(village):
    food, water, materials, population = (
        village["resources"].get("food", 0),
        village["resources"].get("water", 0),
        village["resources"].get("materials", 0),
        village["population"],
    )
    return (
        FOOD_WEIGHT * (food / population)
        + WATER_WEIGHT * (water / population)
        + MATERIALS_WEIGHT * (materials / population)
    )


def main():
    for village_key, village in villages.items():
        happiness = calculate_happiness(village)
        print(f"{village['name']} Happiness: {happiness:.2f}")


if __name__ == "__main__":
    main()
