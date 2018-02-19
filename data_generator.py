import argparse
import sys
import numpy as np
import modules.pyplot_2d as plt



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", metavar="NUM_DATA", help="number of plots to generate", type=int, default=300)
    parser.add_argument("-s", "--shift", type=float, default=2)
    parser.add_argument("-p", "--pattern", choices=['diag', 'left-right', 'one-three'], default='diag')
    parser.add_argument("-l", "--label2", type=int, choices=[0, -1], default=0)
    parser.add_argument("-o", "--output")
    parser.add_argument("-v", "--visualize", action="store_true")
    args = parser.parse_args()

    # generate random data
    quadrant1 = np.array([np.random.randn(args.n) + args.shift, np.random.randn(args.n) + args.shift])
    quadrant2 = np.array([np.random.randn(args.n) - args.shift, np.random.randn(args.n) + args.shift])
    quadrant3 = np.array([np.random.randn(args.n) - args.shift, np.random.randn(args.n) - args.shift])
    quadrant4 = np.array([np.random.randn(args.n) + args.shift, np.random.randn(args.n) - args.shift])

    if args.pattern == 'diag':
        x1 = np.hstack([quadrant4])
        x2 = np.hstack([quadrant2])
    elif args.pattern == 'left-right':
        x1 = np.hstack([quadrant1, quadrant4])
        x2 = np.hstack([quadrant2, quadrant3])
    elif args.pattern == 'one-three':
        x1 = np.hstack([quadrant4])
        x2 = np.hstack([quadrant1, quadrant2, quadrant3])
    else:
        exit("unkown pattern: " + pattern)

    y1 = np.full(args.n, 1, dtype=np.int64)
    y2 = np.full(args.n, args.label2, dtype=np.int64)

    xdata = np.hstack([x1, x2])
    ydata = np.hstack([y1, y2])

    # output target
    fout = open(args.output, 'w') if args.output is not None else sys.stdout

    print("\n".join(["{}\t{}\t{}".format(x0,x1,y) for x0,x1,y in zip(xdata[0], xdata[1], ydata)]), file=fout)

    if args.visualize:
        plt.plot_main(x1, x2)
        plt.show()

    print("done")
