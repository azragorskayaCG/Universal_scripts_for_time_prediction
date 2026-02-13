%%time
from pred_ML import *


"""Set these parameters before launching the code"""

dirname = '/Users/clemence/Documents/Personnal/tim_actions/'

filename = 'BTCUSDT_20k.csv' #BIRBUSDT_020526, BTCUSDT_020526, RIVERUSDT_020526, XAUTUSDT_020526 ok

feature_names = ['start','open','high','low','volume','turnover']


target = 'close'

# dirname = '/Users/clemence/Documents/Аспирантура_наука/1. Работа/2. Нейронные сети/g11-15_solar_cycle24/data/'

# filename = 'dataset.csv'

# feature_names = ['kp', 'V',  'symh', 'Bz', 'P', '08MeV', 'ionT', 'Bt', 'n', 'Rss']

# target = '2MeV'

ahead = 10 #days ahead to predict
offset = 30 #offset days as input
size_split_valtest = 0.2 #set the train and val/test size 
size_split_test = 0.5 #set the val and test size

# =========== 

results = Prediction_ML(dirname, filename, feature_names, target, ahead, offset, size_split_valtest, size_split_test)
ress = results.metrics() #prepare_data / split / features / metrics



data = results.prepare_data()
plt.scatter(data.index, data[target])
plt.xticks(rotation = 45)
plt.ylabel('close')
plt.xlabel('Time')
plt.show()



res = results.predict()
day = 10

plt.plot(results.y_test.index, results.y_test[f'target_day_{day}'], label = 'real')
plt.plot(results.y_test.index-pd.Timedelta(minutes=day), res[day-1], label = 'predicted')
plt.xticks(rotation=45)
plt.ylabel('close')
plt.xlabel('Time')
plt.legend(title = f'Minute {day}')
plt.show()


