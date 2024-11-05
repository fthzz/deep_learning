import os


def rename_files(directory, search, replace):
    """
    在指定目录中查找文件名包含特定字符串的文件，并替换这些字符串。

    :param directory: 要搜索和重命名文件的目录路径
    :param search: 需要被替换的字符串
    :param replace: 替换成的新字符串
    """
    # 确保目录存在
    if not os.path.exists(directory):
        print(f"指定的目录 {directory} 不存在。")
        return
    # i=80
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # replace=str(i)
        # 检查文件名是否包含需要被替换的字符串
        if search in filename:
            # 构造新文件名
            new_filename = filename.replace(search, replace)
            # 构造完整的文件路径
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            # 重命名文件
            os.rename(old_file, new_file)
            print(f"文件已重命名: {filename} -> {new_filename}")
            # i+=1


# 使用示例

directory_path = '/home/fthzzz/Desktop/2110941_1716526530/labels' #你的目录路径
search_string = 'labels'  # 你想要替换的旧字符串
replace_string = ''  # 你想要替换成的新字符串
rename_files(directory_path, search_string, replace_string)