import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n1", help="ResNet depth for Features Network", default=9)
parser.add_argument("-n2", help="ResNet depth for Features Network", default=3)
args = parser.parse_args()

print(args.n1)
print(args.n2)
