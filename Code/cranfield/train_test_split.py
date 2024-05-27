import json

# Load the original JSON file
with open('cranfield/cran_queries.json', 'r') as f:
    data = json.load(f)

# Create the 9 new JSON files
file_ranges = []

for start in range(1, 218, 9):
    end = min(start + 9, 226)  # Ensure the end doesn't exceed 226
    file_ranges.append(range(start, end))


for i, ranges in enumerate(file_ranges, start=1):
    new_data = [query for query in data if query['query number'] in ranges]
    with open(f'cranfield/cran_queries_{i}.json', 'w') as f:
        json.dump(new_data, f, indent=2)

        