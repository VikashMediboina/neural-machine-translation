from datasets import load_dataset
import json
# Load the WMT15 de-en translation dataset
wmt15_de_en_dataset = load_dataset('wmt15', 'de-en')

# Accessing the training split
train_data = wmt15_de_en_dataset['train']
test_data = wmt15_de_en_dataset['test']
validation_data = wmt15_de_en_dataset['validation']
# wmt15_de_en_dataset.to_json('data.json')
train_data.to_json('train.json')
test_data.to_json('test.json')
validation_data.to_json('validation.json')
with open("data.json", "w") as f:
  json.dump(wmt15_de_en_dataset, f, indent=4)
# Accessing a specific example in the training data
example = train_data[0]
print(len(train_data))

# with open("train.json", "w") as f:
#   json.dump({"data":train_data}, f, indent=4)

# with open("test.json", "w") as f:
#   json.dump({"data":test_data}, f, indent=4)
# with open("validation.json", "w") as f:
#   json.dump({"data":test_data}, f, indent=4)