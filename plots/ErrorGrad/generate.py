# 定义要读取的文件名
# filename = 'files.txt'
# filename = 'llms.txt'
filename = 'burst.txt'
# filename = 'all_cv.txt'
# filename = 'LLM-sgd.txt'


# 创建一个空列表，用来存储结果
result = []

# parts = ["part1", "part2"]  # Example list
# formatted_string = f'{""}, "{parts[1]}"'
# print(formatted_string)

new_py_codes = []
# 读取文件
with open(filename, 'r', encoding='utf-8') as file:
    # 遍历每一行
    for line in file:
        # 去掉行首尾的空白符并分割成两个部分
        parts = line.strip().split()
        # 将分割的部分添加到结果列表中
        if len(parts) == 2:  # 确保每行有两个部分
            result.append(parts)
            py_code = f'build_run("hpml-hkbu/DDP-Train/{parts[1]}", CIFAR10_RES18,\n' + \
                    '{' + '"": ""' + '}' + f', "{parts[0]}")'
                    # f'{"": ""}, "{parts[1]}")'

            new_py_codes.append(py_code)


# 打印结果
print(result)


# 将列表保存到 txt 文件
with open('burst.py', 'w') as file:
    for item in new_py_codes:
        file.write(f"{item}\n")  # 每个元素写在一行
















