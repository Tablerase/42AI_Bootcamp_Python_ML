import sys
from random import randint

def print_game_intro():
    print(
        """This is an interactive guessing game!
You have to enter a number between 1 and 99 to find out the secret number.
Type 'exit' to end the game.
Good luck!
"""
    )

def guess():
    to_find : int = randint(1, 99)
    tries_count : int = 1
    while True:
        input_number : int = 0
        try:
            input_value = input("What's your guess between 1 and 99?\n")
            if input_value == 'exit':
                print("Goodbye!")
                break
            input_number = int(input_value)
            assert isinstance(input_number, int)
            if input_number < to_find:
                print("Too low!")
            elif input_number > to_find:
                print("Too hight!")
            elif input_number == to_find:
                print("Congratulations, you've got it!")
                if to_find == 42:
                    print("The answer to the ultimate question of life, the universe and everything is 42.")
                if tries_count == 1:
                    print("Amazing, first try!")
                else:
                    print(f"You won in {tries_count} attempts!")
                break
            else: 
                print("Something weird happen!", file=sys.stderr)
            tries_count += 1
        except (EOFError, KeyboardInterrupt):
            print("Goodbye!")
            break
        except (AssertionError, ValueError):
            print("That's not a number")
            continue


if __name__ == "__main__":
    print_game_intro()
    guess()