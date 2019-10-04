import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", help="type for the resnet", default=5)
args = parser.parse_args()

print(args.n)
