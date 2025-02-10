kata = {
'Python': 'Guido van Rossum',
'Ruby': 'Yukihiro Matsumoto',
'PHP': 'Rasmus Lerdorf',
}

if __name__ == "__main__":
    for item in kata:
        print(f"{item} created by {kata[item]}")