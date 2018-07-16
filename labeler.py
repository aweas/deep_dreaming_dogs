def label(x):
    if 'cat' in x:
        return x.strip() + '; 0\n'
    elif 'dog' in x:
        return x.strip() + '; 1\n'
    else:
        return x.strip() + '; 2\n'


with open('new_files_list.txt') as f:
    files_list = f.readlines()
    labeled = list(map(label, files_list))

    with open('labeled.csv', 'w') as fw:
        for line in labeled:
            fw.write(line)
