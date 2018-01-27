import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=300)
    parser.add_argument("-d", "--data-pattern", choices=['diag', 'left-right', 'one-others'], default='diag')
    parser.add_argument("-l", "--label2-negative", action="store_true")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    # generate random data
    quadrant1 = np.array([np.random.randn(args.n) + 2, np.random.randn(args.n) + 2])
    quadrant2 = np.array([np.random.randn(args.n) - 2, np.random.randn(args.n) + 2])
    quadrant3 = np.array([np.random.randn(args.n) - 2, np.random.randn(args.n) - 2])
    quadrant4 = np.array([np.random.randn(args.n) + 2, np.random.randn(args.n) - 2])

    if args.data_pattern == 'diag':
        x1 = np.hstack([quadrant4])
        x2 = np.hstack([quadrant2])
    elif args.data_pattern == 'left-right':
        x1 = np.hstack([quadrant1, quadrant4])
        x2 = np.hstack([quadrant2, quadrant3])
    elif args.data_pattern == 'one-others':
        x1 = np.hstack([quadrant4])
        x2 = np.hstack([quadrant1, quadrant2, quadrant3])
    else:
        exit("unkown pattern: " + pattern)

    y1 = np.full(args.n, 1, dtype=np.int64)
    label2 = -1 if args.label2_negative else 0
    y2 = np.full(args.n, label2, dtype=np.int64)

    xdata = np.hstack([x1, x2])
    ydata = np.hstack([y1, y2])


    print("".join([ "{}\t{}\t{}\n".format(x0,x1,y) for x0,x1,y in zip(xdata[0], xdata[1], ydata)]), end="")


