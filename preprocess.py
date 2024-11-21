import os
import json
import csv
import random
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def change_xlsx_to_csv(excel_file, csv_file_path):
    df = pd.read_excel(excel_file)

    df.to_csv(csv_file_path, index=False)


def filter_data(csv_file_path):
    data = []
    with open(csv_file_path, mode='r', encoding='latin1') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if "No smiles found for this drug" in row['Smiles_1'] or "No smiles found for this drug" in row['Smiles_2']:
                continue
            assert row['Smiles_1'] != row['Smiles_2'], "Smiles_1 and Smiles_2 are the same."
            assert "No smiles found for this drug" not in row['Smiles_1'], "Smiles_1 is not found."
            assert "No smiles found for this drug" not in row['Smiles_2'], "Smiles_2 is not found."
            data.append(row)
    return data

# def preprocess_data(data):
#     data_dict = {}
#     for row in data:
#         key = f"{row['Smiles_1']}|{row['Smiles_2']}"
#         data_dict[key] = [[f"The drug interactions are {row['interaction_type'].lower()}. {row['description']}"]]
#     return data_dict

# def preprocess_data(data):
#     processed_data = []
#     for row in data:
#         processed_data.append({
#             "smiles_1": row['Smiles_1'],
#             "smiles_2": row['Smiles_2'],
#             "interaction_type": row['interaction_type'].lower()
#         })
#     return processed_data

def preprocess_data(data):
    processed_data = {}
    for row in data:
        key = f"{row['Smiles_1']}|{row['Smiles_2']}"
        processed_data[key] = [[row['interaction_type'].lower()]]
    return processed_data


def split_dataset(data, train_ratio=0.8, val_ratio=0.1, seed=42):
    assert train_ratio + val_ratio < 1.0, "train_ratio + val_ratio must be less than 1.0"

    total_count = len(data)

    # Shuffle data for random split
    random.seed(seed)
    shuffled_data = list(data.items())
    random.shuffle(shuffled_data)
    
    # Calculate sizes
    train_size = int(total_count * train_ratio)
    val_size = int(total_count * val_ratio)
    test_size = total_count - train_size - val_size

    assert train_size + val_size + test_size == total_count, "Train, Val, and Test sizes do not sum up to total count."

    # Split data
    train_data = dict(shuffled_data[:train_size])
    val_data = dict(shuffled_data[train_size:train_size + val_size])
    test_data = dict(shuffled_data[train_size + val_size:])

    print(f"Train data ratio: {len(train_data) / total_count * 100:.2f}%")
    print(f"Val data ratio: {len(val_data) / total_count * 100:.2f}%")
    print(f"Test data ratio: {len(test_data) / total_count * 100:.2f}%")

    return train_data, val_data, test_data

def main():
    excel_file = os.path.join(CURRENT_DIR, "data", "new_drug_drug_interaction.xlsx")
    csv_file_path = f"{excel_file.split('.')[0]}.csv"
    
    if not os.path.exists(csv_file_path):
        change_xlsx_to_csv(excel_file, csv_file_path)
    
    filtered_data = filter_data(csv_file_path)
    
    processed_data = preprocess_data(filtered_data)
    
    train_data, val_data, test_data = split_dataset(processed_data)


    with open(os.path.join(CURRENT_DIR, 'data','drug_drug_interaction_train.json'), 'w') as f:
        json.dump(train_data, f, indent=4)
    with open(os.path.join(CURRENT_DIR, 'data', 'drug_drug_interaction_val.json'), 'w') as f:
        json.dump(val_data, f, indent=4)
    with open(os.path.join(CURRENT_DIR, 'data', 'drug_drug_interaction_test.json'), 'w') as f:
        json.dump(test_data, f, indent=4)

if __name__ == "__main__":
    main()
