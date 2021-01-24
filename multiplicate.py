def multiplicate(input_list):
    total_prod = 1
    for a in input_list:
        total_prod *= a

    output_list = list()
    for a in input_list:
        output_list.append(total_prod/a)

    return output_list


if __name__ == '__main__':
    import sys
    input_list = [int(x) for x in sys.argv[1][1:-1].split(',')]
    output_list = multiplicate(input_list)
    print(output_list)
