import os
import glob

# UPDATE THESE PATHS to point to your actual label folders
label_folders = [
    "label/train",
    "label/val" 
]

for folder in label_folders:
    txt_files = glob.glob(os.path.join(folder, "*.txt"))
    files_modified = 0
    
    for file_path in txt_files:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
        modified = False
        new_lines = []
        
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
                
            # 1. Merge Half Helmets (3) into With Helmet (2)
            if parts[0] == '3':
                parts[0] = '2'
                modified = True
                
            # 2. Shift Rider (4) down to (3) to keep classes contiguous
            elif parts[0] == '4':
                parts[0] = '3'
                modified = True
                
            # Rebuild the line and add to our new list
            new_lines.append(" ".join(parts) + "\n")
                
        # Overwrite the original file securely
        if modified:
            with open(file_path, 'w') as file:
                file.writelines(new_lines)
            files_modified += 1
            
    print(f"Scanned {folder}: Successfully updated {files_modified} files.")

print("Dataset standardization complete!")