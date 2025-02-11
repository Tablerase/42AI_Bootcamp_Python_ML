from typing import List

"""

https://docs.python.org/3.9/library/functions.html?highlight=property#property
"""

class Recipe:
    # Attributes

    def __init__(self, in_name: str, in_cook_lvl: int, in_cook_time: int,
                 in_ingredients: List[str], in_recipe_type: str, in_description: str = ""):
        self.name = in_name
        self._cooking_lvl = None
        self.cooking_lvl = in_cook_lvl
        self._cooking_time = None
        self.cooking_time = in_cook_time
        self.ingredients = in_ingredients
        self._recipe_type = None
        self.recipe_type = in_recipe_type
        self.description = in_description

    @property
    def cooking_lvl(self):
        return self._cooking_lvl
    @cooking_lvl.setter
    def cooking_lvl(self, value : int):
        if not (1 <= value <= 5):
            raise ValueError(f"cooking lvl value should be in range 1 to 5, got {value}")
        self._cooking_lvl = value

    @property
    def cooking_time(self):
        return self._cooking_time
    @cooking_time.setter
    def cooking_time(self, value : int):
        if value < 0:
            raise ValueError(f"time only positive numbers, got {value}")
        self._cooking_time = value

    @property
    def recipe_type(self):
        return self._recipe_type
    @recipe_type.setter
    def recipe_type(self, value: str):
        allowed_types =  ['starter', 'lunch', 'dessert']
        if not (value in allowed_types):
            raise ValueError(f"recipe type as to be {allowed_types}, got {value}")
        self._recipe_type = value


    def __str__(self):
        return f"""_______[Recipe for {self.name}]_______
    Cooking_lvl: {'':>10}{self.cooking_lvl}
    Cooking_time: {'':>10}{self.cooking_time} min
    Ingredients: {'':>10}{self.ingredients}
    Recipe type: {'':>10}{self.recipe_type}
    Description: {'':>10}{self.description}"""


