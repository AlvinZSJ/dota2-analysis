# map heroes names to csv file
# directory should be changed
import json
def loadFont():
    f = open("C:/Users/dell/Desktop/heroes.json", encoding='utf-8')
    setting = json.load(f)
    return setting['heroes']


t = loadFont()
# print(t[0])
# print(t[0]['id'])
def file_process(t):
    names=[]
    for i in range(114):
        names.append('')
    for i in range(len(t)):
        names[t[i]['id']]=t[i]['name']
    # print(names)

    # add 3 col names before
    names.insert(0,'winner')
    names.insert(1,'cluster_id')
    names.insert(2,'game_mode')
    names[3]='game_type'
    return names
    # print(names)
names=file_process(t)
import pandas as pd
df=pd.read_csv('C:/Users/dell/Desktop/5012/dota2Train.csv', header=None, names =names )
df.to_csv('C:/Users/dell/Desktop/5012/nTrain.csv',index=False)
df2=pd.read_csv('C:/Users/dell/Desktop/5012/dota2Test.csv', header=None, names =names )
df2.to_csv('C:/Users/dell/Desktop/5012/nTest.csv',index=False)