import pandas as pd
from pandas import DataFrame
import logging

logger = logging.getLogger('jsonml')


class MDataFrame:
    '''
    基于DataFrame 的数据处理类，构造函数只接收DataFrame一个参数
    '''
    def __init__(self, data=None):
        if isinstance(data, DataFrame):
            self.data = data
        else:
            raise Exception("error data type, must be DataFrame")

    def columns(self):
        '''
        所有列名
        :return: list 列名
        '''
        return self.data.columns.values.tolist()

    def size_column(self):
        '''
        列的长度
        :return: int 列数
        '''
        return len(self.data.columns)

    def size_row(self):
        '''
        数据行数
        :return: int 行数
        '''
        return len(self.data)

    def select(self, columns):
        '''
        选取部分列
        :param columns: list 所选列名
        :return: 在当前对象上选取部分列，与当前对象一致
        '''
        self.data = self.data[[column for column in columns]]
        return self

    def drop(self, columns):
        '''
        去掉部分列
        :param columns: list 删除列名
        :return: 在当前对象上删除部分列，与当前对象一致
        '''
        self.data.drop(columns, axis=1, inplace=True)
        return self

    def copy_column(self, input_columns, output_columns):
        '''
        复制列
        :param input_columns:
        :param output_columns:
        :return: 在当前对象上复制列，与当前对象一致
        '''
        for index, input in enumerate(input_columns):
            self.data[output_columns[index]] = self.data[input]
        return self

    def add_column(self, columns, value):
        '''
        添加列
        :param columns: 列名，list or str
        :param value: 添加列填充的值
        :return:
        '''
        if isinstance(columns, (list, set)):
            for column in columns:
                self.data[column] = value
        else:
            self.data[columns] = value

    def merge(self, mDataFrame):
        '''
        与另外一个对象进行逐行合并，等效于pandas的concat([], axis=1)
        :param MDataFrame: 列名与当前对象不能重复，否则会报错，行数需与当前对象相同，否则报错
        :return: 合并后的对象，与当前对象一致
        '''
        self.data = pd.concat([self.data, mDataFrame.datas()], axis=1)
        return self

    def join(self, MDataFrame, columns, type='inner'):
        '''
        与另外一个对象进行按行叉乘合并
        :param MDataFrame: 列名与当前对象仅不能重复，否则会报错，行数需与当前对象相同，否则报错
        :param columns: join列名
        :param type: join类型，可以选择 'inner', 'outer', 'left_outer', 'right_outer'
        :return: 合并后的对象，与当前对象一致
        '''
        return self

    def rename(self, columns):
        '''
        重命名列名
        :param columns: 支持dict和list。 list长度需与当前列长一致，dict时新列名不能为空
        :return: 修改列名后的对象，与原来对象是同一个
        '''
        if isinstance(columns, dict):
            self.data.rename(columns=columns, inplace=True)
        else:
            self.data.columns = columns
        return self

    def order_column(self, columns):
        '''
        对列按照指定顺序输出
        :param columns: 指定顺序
        :return: 修改列顺序后的对象，与原来对象是同一个
        '''
        self.data = self.data[columns]
        return self

    def process_udf(self, udf, columns_input, columns_output=None, keep_input_columns=False):
        '''
        单行多列处理，包括单列变单列，单列表多列，多列变单列，多列变多列
        :param udf: udf: 自定义处理方法的对象
        :param columns_input: list[list] 输入列名, 第二维表示用该udf可以同时处理多个，第一维需与udf方法参数顺序和个数一致
        :param columns_output: list[list] 输出列名，可为空，为空时默认填充。如果第一维大小为1，并且udf输出为list，则展开list，并自动加后缀命名
        :param keep_input_colums: boolean 是否保留输入列，默认不保留
        :return: 数据处理后的对象，与原来对象是同一个
        '''
        if len(columns_input) == 0:
            raise Exception('columns can not be empty')

        # 一维数组处理
        if not isinstance(columns_input[0], list):
            columns_tmp = []
            for columns in columns_input:
                columns_tmp.append([columns])
            columns_input = columns_tmp
        if not isinstance(columns_output[0], list):
            columns_tmp = []
            for columns in columns_output:
                columns_tmp.append([columns])
            columns_output = columns_tmp

        for index, columns in enumerate(columns_input):
            output_column = columns_output[index]
            logger.debug('input = ' + str(columns))
            logger.debug('output = ' + str(output_column))
            result_list = [udf.process(*x) for x in zip(self.data[columns[0]])] if len(columns) == 1 \
                else [udf.process(*x) for x in zip(*tuple([self.data[column] for column in columns]))]
            if not keep_input_columns:
                self.drop(columns)
            else:
                self.drop([column for column in columns if column in output_column])

            if len(output_column) > 1:
                result_df = DataFrame(result_list, columns=output_column)
                self.data = pd.concat([self.data, result_df], axis=1)
            else:
                self.data[output_column[0]] = result_list
        return self

    def process_udaf(self, udf, columns_input, columns_output=None):
        '''
        多行多列处理，包括多行合并成一行
        :param udf: 自定义处理方法，或lambda表达式
        :param columns_input: list[list] 输入列名, 第二维表示用该udf可以同时处理多个，第一维需与udf方法参数顺序和个数一致
        :param columns_output: list[list] 输出列名，可为空，为空时默认填充
        :return: 数据处理后的对象，与原来对象是同一个
        '''
        if len(columns_input) == 0:
            raise Exception('columns can not be empty')

        # 一维数组处理
        if not isinstance(columns_input[0], list):
            columns_tmp = []
            for columns in columns_input:
                columns_tmp.append([columns])
            columns_input = columns_tmp
        if not isinstance(columns_output[0], list):
            columns_tmp = []
            for columns in columns_output:
                columns_tmp.append([columns])
            columns_output = columns_tmp
            # 一维数组处理

        results = []
        for index, columns in enumerate(columns_input):
            if isinstance(columns, list) and len(columns) == 1:
                input_data = [self.data[columns[0]].values.tolist()]
            else:
                input_data = self.data[columns].values.tolist()
            result = udf.process(input_data)

            results.append(result)
        return results

    def process_udtf(self, udf, columns_input, columns_output=None):
        '''
        多行多列处理，包括单列变单列，单列表多列，多列变单列，多列变多列
        :param udf: 自定义处理方法，或lambda表达式
        :param columns_input: list[list] 输入列名, 第二维表示用该udf可以同时处理多个，第一维需与udf方法参数顺序和个数一致
        :param columns_output: list[list] 输出列名，可为空，为空时默认填充
        :return: 数据处理后的对象，与原来对象是同一个
        '''
        return self

    def filter(self, udf, columns):
        '''
        过滤部分行
        :param udf: 自定义过滤方法，或lambda表达式
        :param columns: list udf中需要操作的列名
        :return: 过滤后的的对象，与原来对象是同一个
        '''
        return self

    def distinct(self, colums=None, keep='first'):
        '''
        按照colums 列删除重复行
        :param colums: list 去重列list，默认为整行去重
        :param keep: str 保留行，默认为第一行
        :return: 删除重复行后的数据
        '''
        return self.data.drop_duplicates(colums, keep=keep)

    def distinct_values(self, column=None):
        '''
        获取某列全部值，使用时label较少的情况使用
        :param colum: str 列名
        :return: list 去重后的数据
        '''
        ss = self.data[[column]].drop_duplicates(column).values.tolist()
        return [s[0] for s in ss]

    def group(self, group_colums, agg_udf, agg_colums):
        '''
        按行group
        :param group_colums: list，set, str: 用于group的列名
        :param agg_udf: group后合并列处理方法
        :param agg_colums: list udf所需的列
        :return: 按行合并后的对象，与原来对象是同一个
        '''
        return self

    def fillna(self, value='mean', columns=None):
        '''
        填充默认值
        :param value: 填充方法，包括mean, min, max, 自定义
        :param columns:
        :return: 填充后的对象，与原来对象是同一个
        '''
        return self

    def replace(self, columns, oldvalue, newvalue):
        '''
        数据内容替换
        :param columns: 需要替换的列
        :param oldvalue: 替换前的数据
        :param newvalue:  替换后的数据
        :return: 替换后的对象，与原来对象是同一个
        '''
        return self

    def split(self, percent=0.8, count=-1, split_type='random'):
        '''
        按行拆分数据集合
        :param percent: 拆分比例
        :param count:  拆分行数，小于0不生效，与percent是或的关系，有一条满足立即停止
        :param split_type: 拆分方法，默认随机，可以是 'head', 'tail' 等
        :return: 拆分的两个部分数据
        '''
        return self, MDataFrame(self.data)

    def take(self, count=10, split_type='head'):
        '''
        获取部分数据
        :param count: 数据条数
        :param split_type: 拆分类型 'head', 'tail'， 'random'
        :return: 拆分的第一部分数据
        '''
        self.data = self.data.head(count)
        return self

    def union(self, mDataFrame):
        '''
        与另外一个对象进行逐列合并
        :param mDataFrame: 需与当前对象列名完全一致，否则报错
        :return: 合并后的对象，与当前一致
        '''
        if self.data.empty:
            self.data = mDataFrame.datas()
        else:
            self.data = pd.concat([self.data, mDataFrame.datas()], axis=0, ignore_index=True)
        return self

    def datas(self):
        '''
        获取当前对象的数据集合
        :return: DataFrame
        '''
        return self.data

    def __str__(self):
        '''
        重写该方法，用于print()时正确输出
        :return:
        '''
        return self.data.__str__()

    def print(self):
        '''
        打印数据
        '''
        print(self.data)

    # def onehot_encoder(self, columns, istrain=True, ignore_labels=None, labels=None, suffix='_oh', keep_input_columns=False):
    #     '''
    #     onehot 编码
    #     :param column: list 需要编码的列
    #     :param istrain: 是否为训练模式，训练模式会加载全部label，预测模式的label直接从配置文件加载
    #     :param ignore_labels: list, 不参与编码的label列表，当istrain=True and labels=None 时才生效
    #     :param labels: list 默认为空, 如果输入，则istrain失效，优先使用输入labels
    #     :param suffix: str 自定义输出列后缀
    #     :param keep_input_columns: 是否保留输入列，默认不保留
    #     :return: 编码结果 与原来是同一个对象
    #     '''
    #     if len(columns) == 0:
    #         raise Exception('columns can not be empty')
    #
    #     for column in columns:
    #         if labels:
    #             label = labels[column]
    #         elif istrain:
    #             label = self.distinct_values(column)
    #             if ignore_labels:
    #                 label = list(set(label) - set(ignore_labels))
    #             label.sort()
    #         else:
    #             label = []
    #         udf = OneHot(label)
    #         out_column_name =[column + suffix + '_' + str(index) for index, _ in enumerate(label)]
    #         self.process_udf(udf, [column], out_column_name, keep_input_columns)
    #     return self
    #
    # def moving_window(self, columns, mw_info, keep_input_columns=False):
    #     '''
    #     滑窗编码
    #     :param columns: 需要编码的列
    #     :param mw_info: 窗口信息
    #     :param keep_input_columns: 是否保留原有列
    #     :return: 编码结果 与原来是同一个对象
    #     '''
    #     for column in columns:
    #         if mw_info:
    #             label = mw_info[column]
    #         else:
    #             raise Exception('error mw_info')
    #         udf = Window(label)
    #         out_column_name =[column + '_win_' + str(index) for index, _ in enumerate(label)]
    #         self.process_udf(udf, [column], out_column_name, keep_input_columns)
    #     return self
    #
    # def bucket_encoder(self, columns, bucket_info, keep_input_columns=False):
    #     '''
    #     滑窗编码
    #     :param columns: 需要编码的列
    #     :param mw_info: 窗口信息
    #     :param keep_input_columns: 是否保留原有列
    #     :return: 编码结果 与原来是同一个对象
    #     '''
    #     for column in columns:
    #         if bucket_info:
    #             label = bucket_info[column]
    #         else:
    #             raise Exception('error bucket_info')
    #         udf = Bucket(label)
    #         out_column_name = [column + '_buc_' + str(index) for index, _ in enumerate(label)]
    #         self.process_udf(udf, [column], out_column_name, keep_input_columns)
    #     return self


def split_feature_and_label_df(df, label_columns=["label"]):
    '''
    将数据拆分成特征数据和标签数据
    PARA:      feature_label_df = DataFrame    待拆分的数据
                label_columns = [str]     标签字段
    RETURN:     feature_df = DataFrame    拆分后的特征数据
                label_df = DataFrame      拆分后的标签数据
    '''

    label_df = df[label_columns]
    df.drop(label_columns, axis=1, inplace=True)

    return df, label_df
