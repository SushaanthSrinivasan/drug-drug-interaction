import pandas as pd

file_path = "../data/new_drug_drug_interaction_2.xlsx"
df = pd.read_excel(file_path)

unique_smiles = pd.concat([df['Smiles_1'], df['Smiles_2']]).dropna().unique()

with open("../data/unique_smiles.txt", "w") as f:
    for smiles in unique_smiles:
        f.write(smiles + "\n")

