import os
import re

def remove_cuda_visible_devices_line(directory):
    # Define the pattern to match lines starting with "CUDA_VISIBLE_DEVICES="
    pattern = re.compile(r'^export CUDA_VISIBLE_DEVICES=.*$', re.MULTILINE)

    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Process only .sh files
            if file.endswith('.sh'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()

                # Remove lines matching the pattern
                modified_content = pattern.sub('', content)

                # Write the modified content back to the file
                with open(file_path, 'w') as f:
                    f.write(modified_content)

                print(f"Processed file: {file_path}")

# Specify the directory to process
directory_to_process = 'scripts'
remove_cuda_visible_devices_line(directory_to_process)
