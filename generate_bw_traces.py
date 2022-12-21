import os

def square_wave(path, max_val, min_val, time_len, period=2):
    f = open(path, 'w')
    half_period = period // 2
    bw = None
    time_stamp = 0
    while time_stamp < time_len:
        for i in range(half_period):
            f.write(str(min_val))
            f.write('\n')
            time_stamp += 1
        for i in range(half_period):
            f.write(str(max_val))
            time_stamp += 1
            f.write('\n')
    f.close()


def constant_wave(path, val, time_len):
    f = open(path, 'w')
    time_stamp = 0
    while time_stamp < time_len:
        f.write(str(val))
        f.write('\n')
        time_stamp += 1
    f.close()



def main():
    # path = '../bw_traces/square_wave.txt'
    # square_wave(path, 3, 0.5, 100)

    path = '../bw_traces/constant_wave.txt'
    constant_wave(path, 2.29, 100)



if __name__ == '__main__':
    main()
