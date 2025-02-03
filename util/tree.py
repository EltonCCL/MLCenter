import os

def has_tf_files(path):
    """Check if directory or its subdirectories contain _tf files"""
    for root, _, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[0].endswith('_tf'):
                return True
    return False

def get_visible_items(path):
    """Get only directories that contain _tf files and _tf files themselves"""
    items = []
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path):
            # Only include directory if it contains _tf files
            if has_tf_files(full_path):
                items.append(item)
        elif os.path.isfile(full_path) and os.path.splitext(item)[0].endswith('_tf'):
            items.append(item)
    return sorted(items)

def print_tf_tree(path, prefix=""):
    contents = get_visible_items(path)
    
    for i, item in enumerate(contents):
        full_path = os.path.join(path, item)
        is_last = i == len(contents) - 1
        
        if is_last:
            current_prefix = "└── "
            new_prefix = "    "
        else:
            current_prefix = "├── "
            new_prefix = "│   "
            
        print(f"{prefix}{current_prefix}{item}")
        
        if os.path.isdir(full_path):
            print_tf_tree(full_path, prefix + new_prefix)

if __name__ == "__main__":
    directory_path = "."
    print(f"Directory structure for: {os.path.abspath(directory_path)}")
    print_tf_tree(directory_path)