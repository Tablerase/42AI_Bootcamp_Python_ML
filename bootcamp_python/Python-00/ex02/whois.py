import sys

def whois(inputs):
    try :
        assert len(inputs) < 3, 'more than one argument is provided'
        input = int(inputs[1])
        assert isinstance(input, int), 'argument is not an integer'
        if input == 0:
            print("I'm Zero.")
        elif input % 2:
            print("I'm Odd.")
        else:
            print("I'm Even.")
        
    except AssertionError as e:
        print("AssertionError:", e)
    
    print(inputs)
        

if __name__ == '__main__':
    whois(sys.argv)