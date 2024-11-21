import json

def convert_data(in_file, out_file):
    with open(in_file, 'r') as f:
        original_data = json.load(f)

    new_data = {}

    for key, value in original_data.items():
        # print(value)
        # break
        # interaction_type = value[0][0].split(" ")[-1]
        sentences = value[0][0].split('.')
        first_sentence = sentences[0].strip().lower()
        interaction_type = first_sentence.split()[-1] if first_sentence else ""

        if interaction_type not in ['major', 'moderate', 'minor']:
            print(f'Unexpected interaction type: {interaction_type}')

        new_data[key] = [[interaction_type]]

    # Save the new data to a new JSON file
    with open(out_file, 'w') as f:
        json.dump(new_data, f, indent=4)

files = ['drug_drug_interaction_train_before', 'drug_drug_interaction_val_before', 'drug_drug_interaction_test_before']
for file in files:
    in_file_name = f'./data/{file}.json'
    out_file_name = f'./data/{file[:-7]}.json'
    convert_data(in_file=in_file_name, out_file=out_file_name)