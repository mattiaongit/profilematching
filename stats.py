import sys
python = sys.version_info[0]


if python < 3:
  from numpy import mean, std, median
  stats_functions = [mean, std, median, min, max]
else:
  from statistics import *
  stats_functions = [mean, stdev, median, min, max]




def distribution(v):
  if len(v) < 2:
    v.append(v[0])
  return [stf(v) for stf in stats_functions]
