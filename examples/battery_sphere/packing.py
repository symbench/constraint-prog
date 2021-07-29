import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
import math


class CirclePacking:

    def __init__(self):
        # read optimal cirle packing arrangements and used battery types
        self.N = list()
        self.ratio = list()
        with open('..\\battery\\arrangements.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';', quotechar='|')
            for row in reader:
                self.N.append(int(row['N']))
                self.ratio.append(float(row['ratio']))

    def pack_circle_with_diam(self, big_d, small_d):
        curr_ratio = big_d / small_d
        # find closest ratio that is still feasible
        curr_N = 0
        if big_d >= small_d:
            for i in range(len(self.N)):
                if self.ratio[i] > curr_ratio:
                    curr_N = i
                    break
            if i >= len(self.N)-2:
                return -1
            return curr_N
        else:
            return 0


def draw_circle(img, x, y, r, w):
    assert x >= 0
    assert y >= 0
    assert w >= 0
    border_points = [(i, j) for i in range(img.shape[0]) for j in range(img.shape[1])
                     if (r - w) ** 2 <= (x - i) ** 2 + (y - j) ** 2 <= r ** 2]
    border_points = np.asarray(border_points)
    img[border_points[:, 0], border_points[:, 1]] = 1.0

    return img


# TODO: add linewidth
def draw_rect(img, x, y, w, h):
    border_points = []
    border_points.extend([(i, y) for i in range(x, x + w) if 0 <= i < img.shape[0]])
    border_points.extend([(x, i) for i in range(y, y + h) if 0 <= i < img.shape[1]])
    if y + h < img.shape[1]:
        border_points.extend([(i, y + h) for i in range(x, x + w) if 0 <= i < img.shape[0]])
    if x + w < img.shape[0]:
        border_points.extend([(x + w, i) for i in range(y, y + h + 1) if 0 <= i < img.shape[1]])

    border_points = np.asarray(border_points)
    img[border_points[:, 0], border_points[:, 1]] = 1.0
    return img


def get_intersection_point_at_dist(x, r):
    return math.sqrt(r ** 2 - x ** 2)


def get_rect_at_dist(x, r, cx, cy):
    y = int(get_intersection_point_at_dist(x, r))
    return cy - y, cx - x, 2 * y, 2 * x


if __name__ == '__main__':
    if '--h' in sys.argv or '--help' in sys.argv:
        print("packing.py --h --v --plot --min_r 10 --max_r 500 --step 1")
        print("--v: verbose turned on")
        print("--plot: plotting turned on")
        print("--min_r value: sphere radius list start value [int]")
        print("--max_r value: sphere radius list end value [int]")
        print("--step value: sphere radius list stepping value [int]")
        print("--h: print help")
        sys.exit(0)
    plotting = True if '--plot' in sys.argv else False
    verbose = True if '--v' in sys.argv else False
    min_r = 10 if '--min_r' not in sys.argv else int(sys.argv[sys.argv.index('--min_r')+1])
    max_r = 500 if '--max_r' not in sys.argv else int(sys.argv[sys.argv.index('--max_r')+1])
    step = 1 if '--step' not in sys.argv else int(sys.argv[sys.argv.index('--step')+1])

    batteries = list()
    with open('..\\battery\\batteries.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';', quotechar='|')
        for row in reader:
            batteries.append(row)

    # create a circle packing instance
    circlepacker = CirclePacking()

    with open('sphere_battery_packing.csv', 'w', newline='') as csvfile:
        fieldnames = ['sphere_D', 'batt_MODEL', 'batt_D', 'batt_L', 'total_N', 'total_CAP', 'total_MASS']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';', quotechar='|')
        writer.writeheader()

        for model, b in enumerate(batteries):
            batt_d = float(b['D'])
            batt_l = float(b['L'])
            batt_m = float(b['M'])
            batt_c = float(b['C'])
            if verbose:
                print("Selected battery type: ", b['Name'])
                print("Battery diameter: ", batt_d)
                print("Battery length: ", batt_l)
                print("Battery mass: ", batt_m)
                print("Battery capacity: ", batt_c)

            # helpers for visual outputs
            cx = 500
            cy = 500
            result = []
            for r in range(min_r, max_r+1, step):
                circles = []
                if plotting:
                    img = np.zeros((1000, 1000))
                    img = draw_circle(img, cx, cy, r, 1)
                dist = batt_l/2
                while dist <= r:
                    rect = get_rect_at_dist( dist, r, cx, cy)
                    if plotting:
                        img = draw_rect(img, int(rect[0]), int(rect[1]), int(rect[2]), int(batt_l))
                    circles.append(rect[2])
                    if dist > batt_l/2:
                        if plotting:
                            img = draw_rect(img, int(rect[0]), int(cx + dist - batt_l), int(rect[2]), int(batt_l))
                        circles.append(rect[2])
                    dist += batt_l

                total_N = 0
                for c in circles:
                    curr_N = circlepacker.pack_circle_with_diam(c, batt_d)
                    if curr_N == -1:
                        total_N = 0
                        break
                    total_N += curr_N
                if verbose:
                    print("Sphere radius:",r,"\tTotal # of batteries: ", total_N)
                if plotting:
                    img -= 1.0
                    img *= -1.0
                    plt.imshow(img, cmap="gray")
                    plt.suptitle("Sphere D: " + str(2*r) + "  ;  battery: " + str(batt_l) + "x" + str(batt_d))
                    plt.title("# of batteries: " + str(total_N))
                    plt.show()
                # if we have >0 batteries, calculate total capacity and mass
                if total_N > 0:
                    MASS_total = total_N * batt_m
                    CAP_total = total_N * batt_c
                    writer.writerow({'sphere_D': 2*r / 1000.0,
                                     'batt_MODEL': model,
                                     'batt_D': batt_d / 1000.0,
                                     'batt_L': batt_l / 1000.0,
                                     'total_N': total_N,
                                     'total_CAP': CAP_total / 1000.0,
                                     'total_MASS': MASS_total})
                    result.append( (r, total_N) )

    result = np.asarray(result)
    if plotting:
        plt.plot( result[:,0], result[:,1], 'r' )
        plt.xlabel("Sphere radius")
        plt.ylabel("# of batteries")
        plt.title("Sphere radius VS # of batteries plot")
        plt.suptitle("Battery diameter: "+str(batt_d)+" , battery length: "+str(batt_l))
        plt.show()
