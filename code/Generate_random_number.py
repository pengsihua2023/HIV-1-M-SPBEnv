# Generate random numbers
import random
import csv

# Generate 10 groups of random numbers
num_groups = 10
numbers_per_group = 40
random_numbers = []

for _ in range(num_groups):
    group = random.sample(range(1, 855), numbers_per_group)
    random_numbers.append(group)

# Transpose the list of random numbers
transposed_numbers = list(map(list, zip(*random_numbers)))

# Save to a CSV file
filename = 'suijishu_AY173956.csv'
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(transposed_numbers)

print(f"Random numbers saved to file: {filename}")
