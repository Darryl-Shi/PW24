import os
import subprocess

def execute_bash_scripts(directory_path):

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and filename.endswith(".sh"):
            try:
                subprocess.run(["bash", file_path], check=True)
                print(f"Executed {filename}")
            except subprocess.CalledProcessError as e:
                print(f"Error executing {filename}: {e}")

subdirectory_path = "scripts"
execute_bash_scripts(subdirectory_path)
