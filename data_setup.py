

import pandas as pd
# import the dataset from Sklearn
from sklearn.datasets import load_breast_cancer


# Read the DataFrame, first using the feature data
data = load_breast_cancer() 
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add a target column, and fill it with the target data
df['target'] = data.target

# Show the first five rows
print(df.head())




