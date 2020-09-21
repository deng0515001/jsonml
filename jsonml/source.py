
from pandas import DataFrame
import os
import time
import json
from jsonml import fileutil
from jsonml import datautil
import numpy as np
import logging
import sys

logger = logging.getLogger('jsonml')


def csv(file_path, columns, field_delimiter=',', ignore_first_line=False, ignore_error_line=False, ignore_blank_line=True):
    '''
    读取CSV文件。支持从本地、hdfs、http读取
    :param file_path: str 可以是单个或者多个文件，多个文件按逗号分隔; 也可以是一层或者多层的文件夹; 不可空
    :param columns: list 列名，需与实际列长度一致，可空，为空时按照数字顺序填充或者首行填充
    :param field_delimiter: str 列分隔符，默认为逗号
    :param ignore_first_line: boolean 是否忽略首行，默认不忽略
    :param ignore_error_line: boolean 是否忽略错误行，包括列长与首个有效行不一致的行，默认不忽略，不忽略时数据有错误会抛出异常。
    :param ignore_blank_line: boolean 是否忽略空白行，默认忽略
    :return: DataFrame 二维数据表
    '''
    files = fileutil.get_paths(file_path)

    if field_delimiter.startswith("0x"):
        field_delimiter = chr(int(field_delimiter, base=16))

    data = []
    length = 0
    first_parts = []

    column_names = []

    for file_path in files:
        with open(file_path) as f:
            if ignore_first_line:
                line = f.readline().strip('\n')
                if len(line) > 0:
                    first_parts = line.split(field_delimiter)
            for line in f:
                line = line.strip('\n')
                if ignore_blank_line and not len(line):
                    continue
                parts = line.split(field_delimiter)

                if length == 0:
                    length = len(parts)
                    if length == len(first_parts) and len(columns) == 0:
                        columns = [[part, 'str', ''] for part in first_parts]
                    elif len(columns) < length:
                        for i in range(len(columns), length):
                            columns.append(["value-" + str(i), 'str', ''])
                    column_names = [column[0] for column in columns]

                if len(parts) == length:
                    parse_parts = []
                    for index, part in enumerate(parts):
                        column_type = columns[index][1]
                        column_default = columns[index][2]
                        if column_type == "int":
                            parse_part = datautil.str2int(part, int(column_default))
                        elif column_type == "float":
                            parse_part = datautil.str2float(part, float(column_default))
                        elif column_type == "bool":
                            parse_part = datautil.str2bool(part, bool(column_default))
                        else:
                            parse_part = part
                        parse_parts.append(parse_part)
                    data.append(parse_parts)
                elif not ignore_error_line:
                    error_desc = line + "\nexpect be " + str(length) + " parts, actually is " + str(
                        len(parts)) + " parts"
                    raise Exception(error_desc)
    if len(data) > 0:
        return DataFrame(data, columns=column_names)
    else:
        raise Exception("no data, please check your input path!")


def parquet(file_path, columns=None, field_delimiter=',', ignore_first_line=False, ignore_error_line=False, ignore_blank_line=True):
    '''
    读取parquet文件。支持从本地、hdfs、http读取
    :param file_path: str 可以是单个或者多个文件，多个文件按逗号分隔; 也可以是一层或者多层的文件夹; 不可空
    :param columns: 列名，需与实际列长度一致，可空，为空时按照数字顺序填充
    :param field_delimiter: 列分隔符，默认为逗号
    :param ignore_first_line: 是否忽略首行，默认不忽略
    :param ignore_error_line: 是否忽略错误行，包括列长与首个有效行不一致的行，默认不忽略，不忽略时数据有错误会抛出异常。
    :param ignore_blank_line: 是否忽略空白行，默认忽略
    :return: DataFrame 二维数据表
    '''
    return DataFrame()


def mysql(ip, port, username, password, db_name, table_name):
    '''
    读取mysql数据表数据
    :param ip: IP地址
    :param port: 端口
    :param username: 用户名
    :param password: 密码
    :param db_name: 数据库名
    :param table_name: 数据表名
    :return: DataFrame 二维数据表
    '''
    return DataFrame()


def hbase(table, filter=None):
    '''
    读取hbase数据
    :param table: 表地址+表名
    :param filter: 过滤条件，不过滤则为全表扫描
    :return: DataFrame
    '''
    return DataFrame()


def es(url, table, filter=None):
    '''
    读取es数据
    :param url: es 地址
    :param table: 表名
    :param filter: 过滤条件
    :return: DataFrame
    '''
    return DataFrame()


def kafka(topic, bootstrap_servers, group_id, startup_mode='latest', callback=None, callback_count=1, callback_time=-1):
    '''
    接收kafka数据流，并按条件回调DataFrame
    :param topic: str kafka配置 topic
    :param bootstrap_servers: str kafka配置 bootstrap_servers
    :param group_id: str kafka配置 group_id
    :param startup_mode: str kafka配置 group_id
    :param callback: function 回调处理方法
    :param callback_count: int 按接收条数回调处理，
    :param callback_time: int 按周期回调处理的时间周期，单位：秒
    '''
    pass


def csv_stream(file_path, columns=None, field_delimiter=',', ignore_first_line=False, ignore_error_line=False, ignore_blank_line=True, callback=None, batch_count=10000, batch_key=None):
    '''
    流式读取CSV文件
    :param file_path: str 可以是单个或者多个文件，多个文件按分号分隔; 也可以是一层或者多层的文件夹; 不可空
    :param columns: list 列名，需与实际列长度一致，可空，为空时按照数字顺序填充
    :param field_delimiter: str 列分隔符，默认为逗号
    :param ignore_first_line: boolean 是否忽略首行，默认不忽略
    :param ignore_error_line: boolean 是否忽略错误行，包括列长与首个有效行不一致的行，默认不忽略，不忽略时数据有错误会抛出异常。
    :param ignore_blank_line: boolean 是否忽略空白行，默认忽略
    :param callback: function 回调处理方法
    :param batch_count: int 按接收条数回调处理，
    :param batch_key: list 列名 接收列做为key分组。默认为None，表示每一条为一组。
                在有序输入流时，能保证同一个key的数据不被截断为两个batch
    :return: 无
    '''
    files = fileutil.get_paths(file_path)
    if field_delimiter.startswith("0x"):
        field_delimiter = chr(int(field_delimiter, base=16))

    data = []
    keys_set = set()
    length = 0
    count = 0
    first_parts = []

    batch_key_index = []
    column_names = []

    start = time.time()
    for file_path in files:
        with open(file_path) as f:
            if ignore_first_line:
                line = f.readline().strip('\n')
                if len(line) > 0:
                    first_parts = line.split(field_delimiter)
            for line in f:
                line = line.strip('\n')
                if ignore_blank_line and not len(line):
                    continue
                parts = line.split(field_delimiter)
                if length == 0:
                    length = len(parts)
                    if length == len(first_parts) and len(columns) == 0:
                        columns = [[part, 'str', ''] for part in first_parts]
                    elif len(columns) < length:
                        for i in range(len(columns), length):
                            columns.append(["value-" + str(i), 'str', ''])
                    column_names = [column[0] for column in columns]

                    if batch_count > 0 and batch_key is not None:
                        columns_index_dict = {column: index for index, column in enumerate(column_names)}
                        batch_key_index = [columns_index_dict[column] for column in batch_key if column in columns_index_dict]

                if len(parts) == length:
                    parse_parts = []
                    for index, part in enumerate(parts):
                        column_type = columns[index][1]
                        column_default = columns[index][2]
                        if column_type == "int":
                            parse_part = datautil.str2int(part, int(column_default))
                        elif column_type == "float":
                            parse_part = datautil.str2float(part, float(column_default))
                        elif column_type == "bool":
                            parse_part = datautil.str2bool(part, bool(column_default))
                        else:
                            parse_part = part
                        parse_parts.append(parse_part)

                    if batch_count > 0 and len(batch_key_index) > 0:
                        key_item = tuple([parse_parts[key] for key in batch_key_index])
                        if key_item not in keys_set:
                            if count >= batch_count:
                                logger.info('batch items len = ' + str(len(data)) + "  cost = " + str(time.time()-start))
                                callback(DataFrame(data, columns=column_names))
                                count -= batch_count
                                data = []
                                keys_set = set()
                                start = time.time()
                            keys_set.add(key_item)
                            count += 1
                    else:
                        if count >= batch_count > 0:
                            logger.info('batch items len = ' + str(len(data)) + "  cost = " + str(time.time() - start))
                            callback(DataFrame(data, columns=column_names))
                            count -= batch_count
                            data = []
                            start = time.time()
                        count += 1
                    data.append(parse_parts)
                elif not ignore_error_line:
                    error_desc = line + "\nexpect be " + str(length) + " parts, actually is " + str(
                        len(parts)) + " parts"
                    raise Exception(error_desc)
    if len(data) > 0:
        logger.info('last batch items len = ' + str(len(data)) + "  cost = " + str(time.time() - start))
        callback(DataFrame(data, columns=column_names))


def stdin_stream(columns=None, field_delimiter=',', ignore_first_line=False, ignore_error_line=False, ignore_blank_line=True, callback=None, batch_count=10000, batch_key=None):
    '''
    流式读取hive text 流
    :param columns: list 列名，需与实际列长度一致，可空，为空时按照数字顺序填充
    :param field_delimiter: str 列分隔符，默认为逗号
    :param ignore_first_line: boolean 是否忽略首行，默认不忽略
    :param ignore_error_line: boolean 是否忽略错误行，包括列长与首个有效行不一致的行，默认不忽略，不忽略时数据有错误会抛出异常。
    :param ignore_blank_line: boolean 是否忽略空白行，默认忽略
    :param callback: function 回调处理方法
    :param batch_count: int 按接收条数回调处理，
    :param batch_key: list 列名 接收列做为key分组。默认为None，表示每一条为一组。
                在有序输入流时，能保证同一个key的数据不被截断为两个batch
    :return: 无
    '''
    if field_delimiter.startswith("0x"):
        field_delimiter = chr(int(field_delimiter, base=16))

    data = []
    keys_set = set()
    length = 0
    count = 0
    first_parts = []
    start = time.time()

    batch_key_index = []
    column_names = []

    lines = 0
    for line in sys.stdin:
        lines += 1
        if ignore_first_line and lines == 1:
            line = line.strip('\n')
            if len(line) > 0:
                first_parts = line.split(field_delimiter)
        else:
            line = line.strip('\n')
            if ignore_blank_line and len(line) == 0:
                continue
            parts = line.split(field_delimiter)
            if length == 0:
                length = len(parts)
                if length == len(first_parts) and len(columns) == 0:
                    columns = [[part, 'str', ''] for part in first_parts]
                elif len(columns) < length:
                    for i in range(len(columns), length):
                        columns.append(["value-" + str(i), 'str', ''])
                column_names = [column[0] for column in columns]

                if batch_count > 0 and batch_key is not None:
                    columns_index_dict = {column: index for index, column in enumerate(column_names)}
                    batch_key_index = [columns_index_dict[column] for column in batch_key if
                                       column in columns_index_dict]

            if len(parts) == length:
                parse_parts = []
                for index, part in enumerate(parts):
                    column_type = columns[index][1]
                    column_default = columns[index][2]
                    if column_type == "int":
                        parse_part = datautil.str2int(part, int(column_default))
                    elif column_type == "float":
                        parse_part = datautil.str2float(part, float(column_default))
                    elif column_type == "bool":
                        parse_part = datautil.str2bool(part, bool(column_default))
                    else:
                        parse_part = part
                    parse_parts.append(parse_part)

                if batch_count > 0 and len(batch_key_index) > 0:
                    key_item = tuple([parse_parts[key] for key in batch_key_index])
                    if key_item not in keys_set:
                        if count >= batch_count:
                            logger.info('batch items len = ' + str(len(data)) + "  cost = " + str(time.time() - start))
                            callback(DataFrame(data, columns=column_names))
                            count -= batch_count
                            data = []
                            keys_set = set()
                            start = time.time()
                        keys_set.add(key_item)
                        count += 1
                else:
                    if count >= batch_count > 0:
                        logger.info('batch items len = ' + str(len(data)) + "  cost = " + str(time.time() - start))
                        callback(DataFrame(data, columns=column_names))
                        count -= batch_count
                        data = []
                        start = time.time()
                    count += 1
                data.append(parse_parts)
            elif not ignore_error_line:
                error_desc = line + "\nexpect be " + str(length) + " parts, actually is " + str(
                    len(parts)) + " parts"
                raise Exception(error_desc)

    if len(data) > 0:
        logger.info('last batch items len = ' + str(len(data)) + "  cost = " + str(time.time() - start))
        callback(DataFrame(data, columns=column_names))


def stdout(df, field_delimiter):
    for _, one_row in df.iterrows():
        outstr = field_delimiter.join('{}'.format(str(item)) for item in one_row)
        sys.stdout.write('{}\n'.format(outstr))


def read_json(file_path):
    '''
    读取一个json文件为字典
    :param file_path: str 文件名
    :return: dict
    '''
    config = None
    if not file_path:
        return config
    if not os.path.exists(file_path):
        print(file_path + " is not exist ")
        return config

    config = json.load(open(file_path))
    return config


def read_csv_to_map(filepath, field_delimiter=',', ignore_first_line=False, name_prefix=""):
    '''
            根据分组文件中的分组信息对数据进行分组, 文件格为csv格式
            PARA:    divede_config_file = str   分组信息文件 文件格为csv格式
                        name_prefix = str    分组类别的前缀
            RETURN:  app_category_map = dict    {app_name: category}
            '''
    if field_delimiter.startswith("0x"):
        field_delimiter = chr(int(field_delimiter, base=16))

    app_category_map = {}
    with open(filepath, 'r') as fp:
        if ignore_first_line:
            head = fp.readline()  # 舍弃 title 行
        for line in fp:
            items = line.strip('\n').split(field_delimiter)
            app_name = items[0]
            category = items[1] if 0 == len(name_prefix) else "_".join([name_prefix, items[1]])
            app_category_map[app_name] = category
    return app_category_map



