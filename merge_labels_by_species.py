import pandas as pd

csv_file = 'concept_labels_per_head.csv'
df = pd.read_csv(csv_file)

def extract_species_id(path):
    fname = path.split('/')[-1]
    parts = fname.split('_')
    species_parts = []
    for part in parts[1:]:
        if part.isdigit():
            break
        species_parts.append(part)
    species_id = '_'.join(species_parts)
    return species_id


df['species_id'] = df['image_path'].apply(extract_species_id)

merged_rows = []
for species_id, group in df.groupby('species_id'):
    row = {}
    row['species_id'] = species_id
    row['image_path'] = group['image_path'].iloc[0]  # keep one example image
    for col in df.columns[1:-1]:  # skip image_path and species_id
        non_na = [str(v) for v in group[col] if pd.notna(v) and v != 'NA']
        unique_labels = sorted(set(non_na))
        if len(non_na) != 0:
            row[col] = ';'.join(unique_labels)
        else:
            row[col] = 'NA'
    merged_rows.append(row)

merged_df = pd.DataFrame(merged_rows)
merged_df.to_csv('concept_labels_per_species_multilabel.csv', index=False)
print('Merged CSV saved as concept_labels_per_species_multilabel.csv')
