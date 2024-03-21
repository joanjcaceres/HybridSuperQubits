import h5py

def load_data_and_metadata(file_path):
    datasets = {}
    metadata = {}

    def load_group(group, path=''):
        # Load attributes as metadata
        for attr in group.attrs:
            metadata_key = f"{path}/{attr}" if path else attr
            metadata[metadata_key] = group.attrs[attr]

        # Load datasets or dive into subgroups
        for name in group:
            item_path = f"{path}/{name}" if path else name
            if isinstance(group[name], h5py.Dataset):
                datasets[item_path] = group[name][...]
            elif isinstance(group[name], h5py.Group):
                load_group(group[name], item_path)

    with h5py.File(file_path, 'r') as file:
        load_group(file)

    return datasets, metadata