import sys
from typing import Dict, List, Tuple

cookbook : Dict[str, Tuple[List[str] , str, int]]

cookbook_struc = ('ingredients', 'meal', 'prep_time')
cookbook = {
    'sandwitch': (['ham', 'bread', 'cheese', 'tomatoes'], 'lunch', 10),
    'cake': (['flour', 'sugar', 'eggs'], 'dessert', 60),
    'salad': (['avocado', 'arugula', 'tomatoes', 'spinach'], 'lunch', 15)
}

def recipe_names():
    print("🍪 Recipes: ")
    for (recipe) in cookbook:
        print(recipe.capitalize())

def recipe_details(recipe : str):
    details = cookbook.get(recipe)
    if details:
        print(
f"""
🍳 Details:
Recipe:\t\t {recipe.capitalize()}
Ingredients: \t{details[0]}
Meal:\t\t {details[1]}
Prep time:\t {details[2]}
"""
        )
    else:
        print(f"{recipe} not found in cookbook")

def recipe_delete(recipe: str):
    action = cookbook.pop(recipe, None)  # Returns None if key doesn't exist instead of raising KeyError
    if action:
        print(f"️🗑️  {recipe} removed from cookbook \n")
    else:
        print(f"🗑️  {recipe} not found in cookbook \n")

def recipe_creation():
    print("🧑‍🍳 New Recipes: ")
    recipe : str = input("Enter a recipe name: \n")
    print("Enter ingredients:")
    ingredients : List[str] = []
    try:
        while True:
            line = input().lower()
            ingredients.append(line)
            if not line:
                break
    except KeyboardInterrupt:
        print("Input canceled by user")
        ingredients = []
        return
    except EOFError:
        print("Input term by EOF")
                
    meal: str = input("Enter a meal type: \n").lower()
    prep_time : int = input("Enter a preparation time: \n").lower()
    if recipe and ingredients and meal and prep_time:
        print(f"Adding {recipe} to cookbook")
        cookbook[recipe] = (ingredients, meal, prep_time)
    else:
        print(f"Recipe data not completely filled")

def print_options():
    print(
    """
    - list :\tList the available recipes
    - detail :\t Give informations about a recipe
    - delete :\t Remove a recipe form cookbook
    - add :\t Add a recipe to the cookbook
    - quit :\t Exit the cookbook
    """)
    


if __name__ == "__main__":
    options : List[str] = ['list', 'detail', 'delete', 'add', 'quit']
    print("""_________[ Cookbook 📓]_________""")
    print("""Select on option to navigate the cookbook:""")
    print_options()
    try:
        while True:
            print("____________________________________________________")
            line = input().lower()
            if not line:
                continue
            if line == options[0]:
                recipe_names()
            elif line == options[1]:
                recipe = input("Please enter a recipe name to get its details: \n").lower()
                recipe_details(recipe)
            elif line == options[2]:
                recipe = input("Please enter a recipe name to delete: \n").lower()
                recipe_delete(recipe)
            elif line == options[3]:
                recipe_creation()
            elif line == options[4]:
                print("Cookbook closed. Goodbye !")
                break;
            else:
                print("Sorry, this option does not exist")
                print_options()
    except KeyboardInterrupt:
        print("Input canceled by user")
    except EOFError:
        print("Input term by EOF")

