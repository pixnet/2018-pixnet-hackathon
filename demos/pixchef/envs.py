import os

PJ = os.path.join
this_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = PJ(this_dir, 'data')
evaluate_dir = PJ(this_dir, 'evaluate')