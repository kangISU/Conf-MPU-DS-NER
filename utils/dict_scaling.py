import random
import os


def random_select(dict_list, size, output_file):
    selected = random.sample(dict_list, size)
    with open(output_file, 'w', encoding='utf-8') as f:
        for word in selected:
            f.writelines(word + '\n')


def select(dict_list, size, output_file):
    selected = dict_list[0:size + 1]
    with open(output_file, 'w', encoding='utf-8') as f:
        for word in selected:
            f.writelines(word + '\n')


def get_dict_list(input_file):
    dict_list = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if len(line.strip()) > 0:
                dict_list.append(line.strip())
    return dict_list


def get_size(dict_lsit, percentage):
    dict_size = len(dict_lsit)
    size = int(dict_size * percentage)
    return size


def main():
    # dataset = 'BC5CDR_Dict_1.0'
    # dict_names = ['Chemical.txt', 'Disease.txt']

    dataset = 'CoNLL2003_Dict_1.0'
    dict_names = ['person.txt', 'location.txt', 'organization.txt', 'misc.txt']

    dictionaries = []
    for d in dict_names:
        path = '../dictionaries/' + dataset + '/' + d
        dictionaries.append(get_dict_list(path))

    percentages = [0.2, 0.4, 0.6, 0.8]
    parent_dir = '../dictionaries'
    count = 1
    for p in percentages:
        count += 1
        directory = dataset + '_' + str(p)
        path = os.path.join(parent_dir, directory)
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        for d, d_n in zip(dictionaries, dict_names):
            size = get_size(d, p)
            output_file = path + '/' + d_n
            # random_select(d, size, output_file)
            select(d, size, output_file)


if __name__ == '__main__':
    main()
