

sizes = [10, 20, 20, 23, 50]
all_combined_sizes = []

print(sizes)
for i in range(len(sizes)):
    s = sizes[i]
    all_combined_sizes.append(s)
    for j in range(i+1, len(sizes)):
        s += sizes[j]
        all_combined_sizes.append(s)
    print(all_combined_sizes)

print(all_combined_sizes)
