import csv


if __name__ == '__main__':
    cmfs = []

    with open('ciexyz31_1.csv') as file_csv:
        reader = csv.reader(file_csv, delimiter=',')

        for row in reader:
            assert len(row) == 4

            cmfs.append([row[0], row[1], row[2], row[3]])

    assert len(cmfs) == 471

    with open('../jaxcolors/color_matching_functions.py', mode='w') as file_python:
        file_python.write('cmfs = [\n')

        for row in cmfs:
            file_python.write('    [')
            file_python.write(', '.join(row))
            file_python.write('],\n')

        file_python.write(']')
