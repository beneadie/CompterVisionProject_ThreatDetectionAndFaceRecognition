import os

# Path to the folder containing YOLO label files
folder_path = "C:/Users/User/Downloads/ALL_FACE_COVERS/valid/labels"
# New class ID to apply to all labels
new_class_id = '0'

# Iterate through all the files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.txt'):  # Process only .txt files
        file_path = os.path.join(folder_path, file_name)

        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Modify the class ID for each line
        updated_lines = []
        for line in lines:
            parts = line.strip().split()
            parts[0] = new_class_id  # Update the class ID
            updated_lines.append(' '.join(parts))

        # Write the updated lines back to the file
        with open(file_path, 'w') as file:
            file.write('\n'.join(updated_lines))

        print(f'Updated {file_name}')
