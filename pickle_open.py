import pandas as pd

 # Replace 'file_path.pkl' with the path to your .pkl file
file_path = "C:\\Projects\\LLM_education\\ds18bb_sol100(vaad).pkl"

df = pd.read_pickle(file_path)

print("Dataframe Loaded Successfully:")
print(df.head())  # Display the first few rows