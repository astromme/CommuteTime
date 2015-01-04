#!/usr/bin/env python
from datetime import datetime, timedelta
import numpy as np

from sklearn.externals import joblib
clf = joblib.load('commute_model.pkl') 

#[u'day_of_month' u'day_of_week' u'direction' u'month_of_year' u'starting_time_hour' u'starting_time_minute' u'year']
dt_now = datetime.now()
now = np.array([[dt_now.month, dt_now.weekday(), 0, dt_now.month, dt_now.hour, dt_now.minute, dt_now.year],
	    		[dt_now.month, dt_now.weekday(), 1, dt_now.month, dt_now.hour, dt_now.minute, dt_now.year]])
now_classifications = clf.predict(now)
print "current commute time to work is predected to be {min} minutes".format(min=now_classifications[0]/60)
print "current commute time to home is predected to be {min} minutes".format(min=now_classifications[1]/60)


def generate_times(t0, step_seconds, num_steps):
	times = [t0]
	delta = timedelta(seconds=step_seconds)
	for i in xrange(num_steps):
		times.append(times[-1]+delta)

	return times

times = generate_times(dt_now, 5*60, 10*24*60/5)

features = np.array([[t.month, t.weekday(), 0, t.month, t.hour, t.minute, t.year] for t in times])
classifications = clf.predict(features)

#for time, classification in zip(times, classifications):
#	print "{0}, {1}".format(time, classification/60.)