import pandas as pd
from sklearn.utils import shuffle

# 1. load the data frame
df = pd.read_csv('data/wdbc.data', header=None)

# 1.1 drop patient-id column
df = df.drop([0], axis=1)

# 2. Normalize the features
for i in range(2,32):
    df[i] = (df[i] - df[i].min()) / (df[i].max() - df[i].min())

# 3. Seperate M, B data
df_M = df[df[1]=='M']
df_B = df[df[1]=='B']

# 3.1 Shuffle the intra-class data
df_M = shuffle(df_M)
df_B = shuffle(df_B)

# 4. Divide data in given ratio for train, test, val
total_M = len(df_M)
train_M = df_M[: int(total_M*0.7)]
val_M   = df_M[int(total_M*0.7) : int(total_M*0.8)]
test_M  = df_M[int(total_M*0.8) :]

total_B = len(df_B)
train_B = df_B[: int(total_B*0.7)]
val_B   = df_B[int(total_B*0.7) : int(total_B*0.8)]
test_B  = df_B[int(total_B*0.8) :]

# 5. Save the diff files for train, test, val data
trainData = pd.concat([train_M, train_B])
valData   = pd.concat([val_M, val_B])
testData  = pd.concat([test_M, test_B])

trainData.to_csv('data/train.data', header=False, index=False)
valData.to_csv('data/val.data', header=False, index=False)
testData.to_csv('data/test.data', header=False, index=False)
