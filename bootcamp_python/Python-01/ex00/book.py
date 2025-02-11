import sys
from typing import List, Dict
from datetime import datetime, time
from recipe import Recipe

"""
https://docs.python.org/3/library/datetime.html
"""

"""
name (str): name of the book,
• last_update (datetime): the date of the last update,
• creation_date (datetime): the creation date of the book,
• recipes_list (dict): a dictionnary with 3 keys: "starter", "lunch", "dessert".
"""


class Book:
    def __init__(self,
                 name: str,
                 last_update: datetime,
                 creation_date: datetime,
                 recipes_list: Dict[str, List[Recipe]]
                 ):
        self.name: str = name
        self.last_update: datetime = last_update
        self.creation_date: datetime = creation_date
        self._recipes_list: Dict[str, List[Recipe]] = {
            'starter': [],
            'lunch': [],
            'dessert': []
        }
        self.recipes_list: Dict[str, List[Recipe]] = recipes_list

    @property
    def recipes_list(self):
        return self._recipes_list

    @recipes_list.setter
    def recipes_list(self, value: Dict[str, List[Recipe]]):
        allowed_types = ['starter', 'lunch', 'dessert']
        for key, values in value.items():
            if not key in allowed_types:
                raise ValueError(
                    f"recipe type as to be {allowed_types}, got {value}")
        self._recipes_list = value

    def get_recipe_by_name(self, name):
        """Prints a recipe with the name \texttt{name} and returns the instance"""
        result_recipe = None
        if self.recipes_list and len(self.recipes_list) > 0:
            for meal, recipes in self.recipes_list.items():
                for recipe in recipes:
                    if recipe.name == name:
                        result_recipe = recipe
        if result_recipe is None:
            print(f"Recipe {name} not found in {self.name}")
        return result_recipe

    def get_recipes_by_types(self, recipe_type):
        """Gets all recipes names for a given recipe_type """
        result_recipes = None
        if self.recipes_list:
            result_recipes = self.recipes_list.get(recipe_type)
        if result_recipes is None:
            print(f"Recipe type {recipe_type} not found in {self.name}")
        return result_recipes

    def add_recipe(self, recipe):
        """Adds a recipe to the book and updates last_update"""
        if not isinstance(recipe, Recipe):
            raise ValueError(f"Add recipe: {recipe} not a Recipe object")
        if self.recipes_list.get(recipe):
            self.recipes_list[recipe.recipe_type].append(recipe)
        else:
            self.recipes_list[recipe.recipe_type] = [recipe]
        self.last_update = datetime.now()

    def __str__(self):
        result = f"{self.name:_^40}\n\n"
        for key, value in self.recipes_list.items():
            result += f"{key.capitalize():^40}\n"
            for recipe in value:
                result += f"{recipe}\n"
        result += f"\n\n{'Good cooking':_^40}\n\n"
        return result
