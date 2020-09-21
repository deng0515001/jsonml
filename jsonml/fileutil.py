import os


def get_fpath_in_dir(dir):
    '''
    获取 directory 目录下所有文件的文件路径
    PARA:   directory = str
    RETURN: file_path = [str]
    '''
    files_path = []
    for home, dirs, files in os.walk(dir):
        for file in files:
            if not file.startswith("."):
                files_path.append(os.path.join(home, file))
    files_path.sort()
    return files_path


def get_paths(file_path, path_delimiter=';'):
    '''
    从一个字符串中解析所有可用文件路径，逗号作为分隔符
    :param file_path: str 可以是单个或者多个文件，多个文件按逗号分隔; 也可以是一层或者多层的文件夹; 不可空
    :param path_delimiter: 多个路径分隔符
    :return: 有序list 文件路径
    '''
    files = []
    for file_path in file_path.strip().split(path_delimiter):
        if os.path.isdir(file_path):
            files.extend(get_fpath_in_dir(file_path))
        elif os.path.isfile(file_path):
            files.append(file_path)
    files.sort()
    return files

