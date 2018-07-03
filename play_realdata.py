import numpy as np
import pandas as pd

data_file = './data/dots_psychophysics.txt'

tdata = pd.read_csv(data_file, 
                       delimiter=' ', skipinitialspace=True, index_col=None, header=None, 
                       names=['coherence', 'direction', 'choice', 'rewarded', 'rt'])
tdata = tdata.values

