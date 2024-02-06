import yaml

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read().replace('---', '')
        return yaml.load(content, Loader=yaml.FullLoader)