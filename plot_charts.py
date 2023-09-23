def combine_paths(path_1, path_2):
    common_states = [state for state in path_1 if state in path_2]
    combined_paths = []
    for common_state in common_states:
        index_1 = path_1.index(common_state)
        index_2 = path_2.index(common_state)
        combined_path_1 = path_1[:index_1] + path_2[index_2:]
        combined_path_2 = path_2[:index_2] + path_1[index_1:]
        if combined_path_1 not in combined_paths:
            combined_paths.append(combined_path_1)
        if combined_path_2 not in combined_paths:
            combined_paths.append(combined_path_2)
    return combined_paths

# # Example usage:
# path_1 = [[0, 0], [0, 1], [1, 1], [2, 1], [1, 1], [0, 1], [0, 2], [0, 3]]
# path_2 = [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2], [2, 3], [1, 3], [0, 3]]

# combined_paths = combine_paths(path_1, path_2)
# for p in combined_paths:
#     print((p))
