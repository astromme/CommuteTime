#!/usr/bin/env python

# stats.py
#   generates and displays ASCII histograms for each feature

import sys
import argparse
from datetime import date, datetime
from collections import defaultdict
from operator import itemgetter, attrgetter, methodcaller

try:
	import ujson as json
except ImportError:
	try:
		import simplejson as json
	except ImportError:
		import json

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', metavar='FILENAME', default='features.json',
                    type=str, nargs='?',
                    help='path to the features file (default is features.json)')
args = parser.parse_args()

try:
	with open(args.input) as f:
		features = json.load(f)
except IOError, e:
	print e
	print "try -h for help"
	sys.exit()
except (ValueError, TypeError), e:
	print "{f} doesn't seem to be in the correct format".format(f=args.input)
	print "try reading the README"
	sys.exit()

if len(features) == 0:
	print "no features found"
	sys.exit()

excluded = [
	"commute_duration_seconds"
]
feature_names = features[0].keys()
histograms = {name:defaultdict(int) for name in feature_names if name not in excluded}

for feature in features:
	for name, histogram in histograms.items():
		histogram[feature[name]] += 1

for name, histogram in histograms.items():
	print name
	max_value = max(histogram.values())
	max_columns = 80
	for value, num_events in sorted(histogram.items(), key=itemgetter(0)):
		print "{0:02d}: {1}".format(value, '*'*int(num_events/float(max_value)*max_columns))
	print ""
