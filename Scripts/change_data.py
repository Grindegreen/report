import numpy as np


def load__and_change_data(file_name):
    data = np.genfromtxt(file_name, names=True)

    print len(data)
    for i in xrange(len(data)):
        data[i][-1] = 0.5
    fin = open('MAXIJ1820p070_daily.txt', 'w')
    for i in xrange(len(data)):
        fin.write(
            str(data[i][0]) + '   ' + str(data[i][-1]) + '    ' + str(data[i][1]) + '    ' + str(data[i][2]) + '\n')
    fin.close()


def main():
    file_name = 'MAXIJ1820p070.lc.txt'
    load__and_change_data(file_name)


main()
