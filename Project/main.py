from pred_ML import *


"""Set these parameters before launching the code"""

dirname = '/Users/clemence/Documents/Аспирантура_наука/1. Работа/2. Нейронные сети/g11-15_solar_cycle24/data/'

filename = 'dataset.csv'

feature_names = ['kp', 'V',  'symh', 'Bz', 'P', '08MeV', 'ionT', 'Bt', 'n', 'Rss']

target = '2MeV'

ahead = 3
offset = 2
size_split_valtest = 0.2
size_split_test = 0.5

# =========== 

results = Prediction_ML(dirname, filename, feature_names, target, ahead, offset, size_split_valtest, size_split_test)
