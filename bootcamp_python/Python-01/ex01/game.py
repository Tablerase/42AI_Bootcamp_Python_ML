import sys

class GotCharacter:
    def __init__(self, first_name : str, is_alive : bool = True):
        self.first_name : str = first_name
        self.is_alive : bool = is_alive

class Mormont(GotCharacter):
    """A Class representing House Mormont
    """
    def __init__(self, first_name = None, is_alive=True):
        super().__init__(first_name=first_name, is_alive=is_alive)
        self.family_name = "Mormont"
        self.house_words = "Here We Stand"

    def print_house_words(self):
        print(self.house_words)

    def die(self):
        self.is_alive = False

if __name__ == "__main__":
    lyanna = Mormont('lyanna')
    print(lyanna.__dict__)

    lyanna.print_house_words()
    print(lyanna.is_alive)
    lyanna.die()
    print(lyanna.is_alive)

    print(lyanna.__doc__)