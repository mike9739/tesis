import pandas as pd 

# data1=pd.read_csv('data/tweets.csv')
# data2=pd.read_csv('data/tweets1.csv')
# data3=pd.read_csv('data/tweets2.csv')

# dataset = pd.concat([data1,data2,data3],join=)

dataset=pd.read_csv('data/Dataset.csv')

usernames=dataset['username'].values
dates = dataset['date'].values
tweets = dataset['tweet'].values


for line in usernames:
    line= line+'\n'
    with open('labels/names.txt','a') as name:
        name.write(line)

for line in tweets:
    line= line+'\n'
    with open('labels/tweets.txt','a') as tweet:
        tweet.write(line)

for line in dates:
    line= line+'\n'
    with open('labels/dates.txt','a') as date:
        date.write(line)


        
    