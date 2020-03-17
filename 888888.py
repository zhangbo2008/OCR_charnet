def load_char_dict(path='datasets/ICDAR2015/test/char_dict.txt', seperator=chr(31)):
    char_dict = dict()
    with open(path, 'rt') as fr:
        for line in fr:
            sp = line.strip('\n').split(seperator)
            char_dict[int(sp[1])] = sp[0].upper()
    return char_dict
print(load_char_dict())