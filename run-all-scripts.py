import os

def execute_bash_scripts(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith(".sh"):
                script_path = os.path.join(root, filename)
                os.system(f"bash {script_path}")

subdirectory_path = "./scripts"
execute_bash_scripts(subdirectory_path)
