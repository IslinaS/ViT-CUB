import os
import csv

concept_dir = '/usr/xtmp/ys298/CUB_200_2011/ece590/concept_dataset'
output_csv = 'concept_labels_per_head.csv'

# Step 1: Collect all high-level → subconcept mappings
all_labels = set()
records = {}

for root, _, files in os.walk(concept_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            folder = os.path.basename(root)
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, '.')

            highlevel = folder.split('::')[0]
            if rel_path not in records:
                records[rel_path] = {}
            records[rel_path][highlevel] = folder

            all_labels.add(folder)

# Step 2: Create per-concept label mapping
highlevel_concepts = sorted({lbl.split('::')[0] for lbl in all_labels})
concept_class_map = {h: [] for h in highlevel_concepts}
for lbl in all_labels:
    h, s = lbl.split('::')
    if s not in concept_class_map[h]:
        concept_class_map[h].append(lbl)

# Step 3: Write CSV
header = ['image_path'] + highlevel_concepts
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    for img, label_dict in records.items():
        row = [img]
        for h in highlevel_concepts:
            row.append(label_dict.get(h, 'NA'))
        writer.writerow(row)

print(f'CSV saved as {output_csv}')