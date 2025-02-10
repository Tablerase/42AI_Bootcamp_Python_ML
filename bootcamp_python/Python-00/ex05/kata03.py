kata = "The right format"

if __name__ == "__main__":
    fill = '-' # Whatever char here
    alignment = '>' # < ^ > : left center righ
    padding = 42 # Amount of padding
    print(f"{kata:{fill}{alignment}{padding}}", end='')
