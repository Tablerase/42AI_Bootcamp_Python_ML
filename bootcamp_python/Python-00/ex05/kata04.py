kata = (0, 4, 132.42222, 10000, 12345.67)

'''
$> python3 kata04.py
module_00, ex_04 : 132.42, 1.00e+04, 1.23e+04
$> python3 kata04.py | cut -c 10,18
,:
'''

if __name__ == "__main__":
    print(
        f"module_{kata[0]:02}, ex_{kata[1]:02} : {kata[2]:.2f}, {kata[3]:.2e}, {kata[4]:.2e}"
    )