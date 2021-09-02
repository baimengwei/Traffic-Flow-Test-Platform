import copy

list_a = ['a', 'b', 'c']
list_b = copy.deepcopy(list_a)

for each in list_a:
    print(each)
    del list_b[0]
    print(list_b)
print(list_b == [])