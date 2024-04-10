import pandas as pd

df=pd.read_csv('pos_neg.csv')

df=df['Sentiment']

l1=[]

for val in df:
    if val==0:
        l1.append(0)
    elif val==0.5:
        l1.append(1)
    else:
        l1.append(2)
        
l1=pd.DataFrame(l1)

l1=pd.concat([df, l1], axis=1)

l1.to_csv('Pos_Neg_Final.csv')


