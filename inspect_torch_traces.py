import os
import json

# this file is used to inspect the traces of the torch model
path_to_traces = ('/home/fespuna/dt01/5/'
                  'gpfs/projects/bsc88/preproduction/'
                  'test_ferran/results/gpt_7b_improved/results'
                  '/12227129/profile_12227129')

jsons = os.listdir(path_to_traces)

first = jsons[0]
assert first.endswith('.pt.trace.json')

with open(os.path.join(path_to_traces, first), 'r') as f:
    data = json.load(f)


print(data.keys())

print(data['traceEvents'][1])



