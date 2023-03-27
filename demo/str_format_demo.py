# basic usage in comparison with old usage
def example1():
    name = 'Fred'
    age = 42
    print("His name is %s and he is %d years old." % (name, age))
    print("His name is {} and he is {} years old.".format(name,age))
    print(f'His name is {name} and he is {age} years old.')
    return

# escape char
def example2():
    student = {'name': 'Yang Song', 'age':31}
    # use both single quote ' and double quote "
    print(f"The student is {student['name']}, he is {student['age']} years old.")
    print(f"{25+6}") # result is 31
    print(f"{{25+6}}") # {25+6}
    print(f"{{{25+6}}}") # {31}
    print(f'Escape using \\ by putting it in front of \'.')
    return

import datetime
# date format
def example4():
    now = datetime.datetime.now()
    print(f'{now:%Y-%m-%d %H:%M}')
    return

def example5():
    val  = 3.14159
    print(f'{val:5.2f} {val:5.5f}')
    print(f'{val:>10.2f}')
    print(f'{val:>10.3f}')
    return


