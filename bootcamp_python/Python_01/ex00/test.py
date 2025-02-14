from datetime import datetime
from book import Book
from recipe import Recipe


if __name__ == "__main__":
    try:
        cake = Recipe("Cake", 5, 20, ["Sugar", "Flour", "Milk"], 'dessert')
        cake2 = Recipe("Cake2", 3, 1, ["Sugar", "Flour", "Milk"], 'dessert', 'Amazing cake with loads of sugar')
        book = Book('Amazing cook', datetime(2025, 2, 10), datetime(2025, 1, 1), {
            'dessert': [cake, cake2],
        })
        print(book)

        find_recipes = book.get_recipes_by_types('dessert')
        print("Find type: ", find_recipes)

        galette = Recipe("Galette", 3, 25, ["Dark flour", "Salt", "Water"], 'lunch', "Yec'hed mat")
        book.add_recipe(galette)

        find = book.get_recipe_by_name('Galette')
        print("Find name: ", find)

    except ValueError as e:
        print(e)
