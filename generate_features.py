#!/usr/bin/env python

# generate_features.py
#   this program takes a LocationHistory.json file exported from Google Takeout
#   and applies a simple heuristic to pull out the commutes in both directions.
#   These commutes are written out as features that can be input for a machine
#   learning algorithm.

from settings import *

#try faster jsons before slower jsons
try:
	import ujson as json
except ImportError:
	try:
		import simplejson as json
	except ImportError:
		import json

#needed for pretty printing the dump
from json import dump as json_slow_dump

from datetime import date, datetime
from collections import defaultdict
from operator import itemgetter, attrgetter, methodcaller
import sys

import argparse

parser = argparse.ArgumentParser(description='Generate features from location history.')
parser.add_argument('-i', '--input', metavar='FILENAME', default='LocationHistory.json',
                    type=str, nargs='?',
                    help='path to the history file (default is LocationHistory.json)')
parser.add_argument('-o', '--output', metavar='FILENAME', default='features.json',
                    type=str, nargs='?',
                    help='path to the output file (default is features.json)')
args = parser.parse_args()

try:
	with open(args.input) as f:
		print "loading location history"
		locations = json.load(f)["locations"]
except IOError, e:
	print e
	print "try -h for help"
	sys.exit()
except (ValueError, TypeError), e:
	print "{f} doesn't seem to be in the correct format".format(f=args.input)
	print "try reading the README"
	sys.exit()

home_latitude_min = home_latitude - home_radius_meters / float(meters_per_degree)
home_latitude_max = home_latitude + home_radius_meters / float(meters_per_degree)
home_longitude_min = home_longitude - home_radius_meters / float(meters_per_degree)
home_longitude_max = home_longitude + home_radius_meters / float(meters_per_degree)

work_latitude_min = work_latitude - work_radius_meters / float(meters_per_degree)
work_latitude_max = work_latitude + work_radius_meters / float(meters_per_degree)
work_longitude_min = work_longitude - work_radius_meters / float(meters_per_degree)
work_longitude_max = work_longitude + work_radius_meters / float(meters_per_degree)

min_commute_seconds = min_commute_minutes * 60
max_commute_seconds = max_commute_minutes * 60

def near_home(location):
	return ((home_latitude_min <= 1./1e7 * location["latitudeE7"] <= home_latitude_max)
	    and (home_longitude_min <= 1./1e7 * location["longitudeE7"] <= home_longitude_max))

def near_work(location):
	return ((work_latitude_min <= 1./1e7 * location["latitudeE7"] <= work_latitude_max)
	    and (work_longitude_min <= 1./1e7 * location["longitudeE7"] <= work_longitude_max))

print "loaded {n} locations".format(n=len(locations))
locations = [l for l in locations if l["accuracy"] <= accuracy_threshold]
print "{n} locations with accuracy <= {accuracy}".format(n=len(locations), accuracy=accuracy_threshold)
locations = [l for l in locations if near_home(l) or near_work(l)]
print "{n} locations near home or work".format(n=len(locations))

# turn [a, b, c, d] of the same location into [a, d]
coalesced = []
for i in xrange(len(locations)):
	# always include the first point
	if i == 0:
		coalesced.append(locations[i])
		continue

	# always include the last point
	if i+1 == len(locations):
		coalesced.append(locations[i])
		continue

	#skip middle locations at home
	if near_home(locations[i-1]) and near_home(locations[i]) and near_home(locations[i+1]):
		continue
	#skip middle locations at work
	if near_work(locations[i-1]) and near_work(locations[i]) and near_work(locations[i+1]):
		continue
	
	coalesced.append(locations[i])

print "{n} coalesced locations".format(n=len(coalesced))

# sort ascending so that we can iterate and extract commutes
coalesced.sort(key=lambda x: int(x["timestampMs"]))

locations_by_day = defaultdict(list)
for location in coalesced:
	location_date = date.fromtimestamp(int(location["timestampMs"])/1000.)
	locations_by_day[location_date.strftime("%Y%m%d")].append(location)

print "{n} days with location".format(n=len(locations_by_day.keys()))

# find commutes to work and to home, placing them in buckets.
home_to_work = []
work_to_home = []
for day, locations in locations_by_day.items():
	for i in xrange(len(locations)-1):
		a = locations[i]
		b = locations[i+1]

		a_time = datetime.fromtimestamp(int(a["timestampMs"])/1000.)
		b_time = datetime.fromtimestamp(int(b["timestampMs"])/1000.)

		if not min_commute_seconds < (b_time - a_time).seconds < max_commute_seconds:
			continue

		if near_home(a) and near_work(b):
			home_to_work.append((a, b))
		elif near_work(a) and near_home(b):
			work_to_home.append((a, b))

print "{n} home to work commutes".format(n=len(home_to_work))
print "{n} work to home commutes".format(n=len(work_to_home))

# returns a feature dict that can be used for training
def feature_from_location_tuple(tuple, direction):
	a, b = tuple

	a_time = datetime.fromtimestamp(int(a["timestampMs"])/1000.)
	b_time = datetime.fromtimestamp(int(b["timestampMs"])/1000.)

	return {
		"direction" : direction,
		"day_of_week" : a_time.weekday(),
		"day_of_month" : a_time.day,
		"month_of_year" : a_time.month,
		"year" : a_time.year,
		"starting_time_hour" : a_time.hour,
		"starting_time_minute" : a_time.minute,
		"commute_duration_seconds" : (b_time-a_time).seconds
	}

# turn commutes into features, mapping direction as an integer (0 and 1)
features  = map(lambda tuple: feature_from_location_tuple(tuple, 0), home_to_work)
features += map(lambda tuple: feature_from_location_tuple(tuple, 1), work_to_home)

try:
	with open(args.output, "w") as f:
		print "writing commute features to {f}".format(f=args.output)
		json_slow_dump(features, f, indent=4)
except IOError, e:
	print e
	print "try -h for help"
	sys.exit()


