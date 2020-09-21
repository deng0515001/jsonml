import json
from jsonml import datesutil
from jsonml import datautil
import logging
import math
from pandas.core.series import Series


logger = logging.getLogger('jsonml')


class Copy:
    def process(self, *x):
        if len(x) == 1:
            return x[0]
        return [xx for xx in x]


class Max:
    def process(self, *x):
        if len(x) == 1 and isinstance(x[0], (list, set)):
            x = x[0]
        if len(x) == 1 and isinstance(x[0], Series):
            x = x[0].values.tolist()
        result = x[0]
        for xx in x:
            result = result if result > xx else xx
        return result


class Min:
    def process(self, *x):
        if len(x) == 1 and isinstance(x[0], (list, set)):
            x = x[0]
        if len(x) == 1 and isinstance(x[0], Series):
            x = x[0].values.tolist()
        result = x[0]
        for xx in x:
            result = result if result < xx else xx
        return result


class Avarage:
    def process(self, *x):
        if len(x) == 1 and isinstance(x[0], (list, set)):
            x = x[0]
        if len(x) == 1 and isinstance(x[0], Series):
            x = x[0].values.tolist()
        total = 0
        for xx in x:
            total += xx
        return total * 1.0 / len(x)


class Sum:
    def process(self, *x):
        if len(x) == 1 and isinstance(x[0], (list, set)):
            x = x[0]
        if len(x) == 1 and isinstance(x[0], Series):
            x = x[0].values.tolist()
        result = 0
        for xx in x:
            result = result + xx
        return result


class Multi:
    def process(self, *x):
        if len(x) == 1 and isinstance(x[0], (list, set)):
            x = x[0]
        if len(x) == 1 and isinstance(x[0], Series):
            x = x[0].values.tolist()
        result = 1
        for xx in x:
            result = result * xx
        return result


class Sub:
    def process(self, x, y):
        return x - y


class Division:
    def process(self, x, y):
        return 0 if y == 0 else x * 1.0 / y


class Count:
    def process(self, *x):
        if len(x) == 1 and isinstance(x[0], (list, set)):
            x = x[0]
        return len(x)


class Replace:
    def __init__(self, source=None, dest=None):
        self.source = source
        self.dest = dest

    def process(self, x):
        if isinstance(self.source, list):
            return x if x not in self.source else self.dest
        return x if x != self.source else self.dest


class ReplaceByColumn:
    def __init__(self, source=None):
        self.source = source

    def process(self, x, dest):
        if isinstance(self.source, list):
            return x if x not in self.source else dest
        return x if x != self.source else dest


class Str2Timestamp:
    def __init__(self, date_type="%Y%m%d"):
        self.date_type = date_type

    def process(self, x):
        result = 0
        try:
            result = datesutil.datetime_to_timestamp(x, self.date_type)
        except:
            logger.info('error datesutil.datetime_to_timestamp:' + str(x))
        return result


class DaysInterval:
    def process(self, x1, x2):
        return datesutil.get_delta_days_by_timestamp(x1, x2)


class StrDaysInterval:
    def __init__(self, format="%Y%m%d"):
        self.format = format

    def process(self, x1, x2):
        result = 0
        try:
            result = datesutil.get_delta_days(x1, x2)
        except:
            logger.info('error datesutil.get_delta_days:' + str(x1) + "\t" + str(x2))
        return result


class SplitToColumns:
    def __init__(self, type=None, output_list=None, default=""):
        self.type = type
        self.output_list = output_list
        self.default = default

    def process(self, x):
        if self.type == "json" or self.type == "dict":
            if self.type == "json":
                try:
                    data = {} if not x or x == "" else json.loads(x)
                except:
                    data = {}
                    logger.info('json.decoder.JSONDecodeError:' + str(x))
            else:
                data = x

            if isinstance(data, dict):
                result = {key: value for key, value in data.items() if key in self.output_list}
            else:
                result = {key: self.default for key in self.output_list}
            res = [result.get(key, self.default) for key in self.output_list]
            return res
        elif self.type == "list":
            if not self.output_list:
                self.output_list = x
            elif len(x) != len(self.output_list):
                logger.info(x)
                logger.info(self.output_list)
                raise Exception("the length of list in the column is not same")
            return x
        else:
            return x


class SplitToList:
    def __init__(self, type=None, split_str=","):
        self.type = type
        self.split_str = split_str

    def process(self, x):
        if "str" == self.type:
            return [] if x == '' else x.split(self.split_str)


class CollectSet:
    def process(self, x):
        if isinstance(x, Series):
            x = x.values
        if isinstance(x[0], (list, set)):
            result = set(item for j in x for item in j)
        else:
            result = set(item for item in x)
        return result


class CollectList:
    def process(self, x):
        if isinstance(x, Series):
            x = x.values
        if isinstance(x[0], (list, set)):
            result = list(item for j in x for item in j)
        else:
            result = list(item for item in x)
        return result


class CollectDict:
    def process(self, x):
        result = {}
        if isinstance(x, Series):
            x = x.values
        if isinstance(x[0], dict):
            for item in x:
                result.update(item)
        else:
            raise Exception('error data for CollectDcit, must be dict')
        return result


class DictSize:
    def __init__(self, filterKeys=[], filterValues=[0]):
        self.filterKeys = filterKeys
        self.filterValues = filterValues

    def process(self, x):
        result = 0
        if isinstance(x, dict):
            if len(self.filterKeys) == 0 and len(self.filterValues) == 0:
                result = len(x)
            else:
                for key, value in x.items():
                    if key not in self.filterKeys and value not in self.filterValues:
                        result += 1
        return result


# class LatestValid:
#     def __init__(self, ascending=True):
#         self.ascending = ascending
#
#     def process(self, x):
#         print('LatestValid')
#         if isinstance(x, Series):
#             print(x.values)
#             result = x.values.last
        # print(x, keys)
        # for key in keys:
        #     if len(x) != len(key):
        #         raise Exception("the keys length and the input length must be the same")
        # last_key = []
        # result = None
        # replace = True
        # if isinstance(x, list):
        #     for index, item in enumerate(x):
        #         if len(last_key) > 0 and item != '':
        #             for index2, key in enumerate(keys):
        #                 if last_key[index2] == key[index]:
        #                     pass
        #                 if last_key[index2] < key[index] and not self.ascending:
        #                     replace = True
        #                 if last_key[index2] > key[index] and self.ascending:
        #                     replace = True
        #                 break
        #         if replace:
        #             result = item
        #             last_key = [key[index] for key in keys]
        #             replace = False
        #     return result
        # else:
        #     raise Exception("invaid params, udaf paras must be list")


class MergeColumn:
    def process(self, *x):
        result = []
        for xx in x:
            if isinstance(xx, list):
                result.extend(xx)
            else:
                result.append(xx)
        return result


class MergeTwoColumnToDict:
    def __init__(self, filterKeys=[], filterValues=[0]):
        self.filterKeys = filterKeys
        self.filterValues = filterValues

    def process(self, x, y):
        if x in self.filterKeys or y in self.filterValues:
            return {}
        return {x: y}


class CountDays:
    def process(self, x):
        if isinstance(x, list):
            result = {}
            for item in x:
                if 't' in item:
                    date = datesutil.timestamp_to_datetime(item['t'], '%Y%m%d')
                    result[date] = result.get(date, 0) + 1
                else:
                    logger.debug("CountByDay error item" + str(item))
            return len(result.keys())
        else:
            return 0


class CountByDay:
    def process(self, x):
        if isinstance(x, list) and len(x) > 0:
            result = {}
            for item in x:
                if 't' in item:
                    date = '19700101'
                    try:
                        date = datesutil.timestamp_to_datetime(item['t'], '%Y%m%d')
                    except:
                        logger.info('error datesutil.timestamp_to_datetime:' + str(x))
                    result[date] = result.get(date, 0) + 1
                else:
                    logger.debug("CountByDay error item" + str(item))
            return result
        else:
            return {}


class CountByKey:
    def process(self, x):
        if isinstance(x, list) and len(x) > 0:
            result = {}
            for xx in x:
                result[xx] = 1 if xx not in result else result[xx] + 1
            return result
        else:
            return {}


# class SumByDay:
#     def process(self, x):
#         if isinstance(x, list) and len(x) > 0:
#             logger.debug('$$$$$$$$$$!!!!!!!!!!!!!!')
#             logger.debug(x)
#             result = {}
#             for item in x:
#                 if 't' in item and 'ext_data' in item and 'during' in item['ext_data']:
#                     date = datesutil.timestamp_to_datetime(item['t'], '%Y%m%d')
#                     result[date] = result.get(date, 0) + datautil.str2float(item['ext_data']['during'])
#                 else:
#                     logger.debug(item)
#             logger.debug(result)
#             return result
#         else:
#             return {}


class SumOnParam:
    def __init__(self, param=None):
        self.param = param

    def process(self, x):
        result = 0
        if isinstance(x, list):
            for item in x:
                if "ext_data" in item:
                    if self.param in item["ext_data"]:
                        result += datautil.str2float(item['ext_data'][self.param], 0)
        else:
            item = x
            if "ext_data" in item:
                if self.param in item["ext_data"]:
                    result += datautil.str2float(item['ext_data'][self.param], 0)
        return result


class SumOnParamByDay:
    def __init__(self, param=None):
        self.param = param

    def process(self, x):
        if isinstance(x, list):
            result = {}
            for item in x:
                if 't' in item and 'ext_data' in item and self.param in item['ext_data']:
                    date = '19700101'
                    try:
                        date = datesutil.timestamp_to_datetime(item['t'], '%Y%m%d')
                    except:
                        logger.info('error datesutil.timestamp_to_datetime:' + str(x))
                    result[date] = result.get(date, 0) + datautil.str2float(item['ext_data'][self.param], 0)
                else:
                    logger.debug("SumOnParamByDay error item: " + str(item))
            return result
        else:
            result = {}
            item = x
            if 't' in item and 'ext_data' in item and self.param in item['ext_data']:
                date = '19700101'
                try:
                    date = datesutil.timestamp_to_datetime(item['t'], '%Y%m%d')
                except:
                    logger.info('error datesutil.timestamp_to_datetime:' + str(x))
                result[date] = result.get(date, 0) + datautil.str2float(item['ext_data'][self.param])
            else:
                logger.debug("SumOnParamByDay error item: " + str(item))
            return result


class BetweenDaysList:
    def __init__(self, format="%Y%m%d", reverse=False):
        self.format = format
        self.reverse = reverse

    def process(self, x1, x2):
        '''
        获取两个日期中间的所有日期
        :param x1: 为较后的日期
        :param x2: 为较早的日期
        :return: list x1与x2之间的日期列表，顺序输出
        '''
        n = datesutil.get_delta_days(x1, x2) + 1
        return datesutil.get_nday_list(x2, n, reverse=self.reverse)


class DictToList:
    def __init__(self, keys, default=0):
        self.default = default
        self.keys = keys

    def process(self, x):
        result = []
        if isinstance(x, dict):
            for day in self.keys:
                result.append(x.get(day, self.default))
        return result


class DictToListByIndex:
    def __init__(self, start_key=0, default=0):
        self.default = default
        self.start_key = start_key

    def process(self, x):
        result = []
        if isinstance(x, dict) and len(x) > 0:
            last_key = max(x.keys())
            for index in range(self.start_key, last_key + 1):
                result.append(x.get(index, self.default))
        return result


class DictToListByColumn:
    def __init__(self, default=0):
        self.default = default

    def process(self, x, keys):
        result = []
        if isinstance(x, dict) and isinstance(keys, list):
            for day in keys:
                result.append(x.get(day, self.default))
        return result


class Mapping:
    def __init__(self, data=None, default=None, type="list"):
        if isinstance(data, dict):
            self.mappings = data
            self.default = default
            self.type = type
        elif isinstance(data, list):
            self.mappings = {value: index for index, value in enumerate(data)}
            self.default = None
            self.type = type

    def process(self, x):
        if isinstance(x, (list, set)):
            return [self.mappings.get(str(item), self.default) for item in x]
        else:
            return self.mappings.get(x, self.default)


class Normalization:
    '''
    自定义UDF 归一化
    '''

    def process(self, x):
        return x


class Standardization:
    '''
    自定义UDF 标准化
    构造函数，参数data，为标准化所需区间参数
    '''

    def __init__(self, data=None):
        self.data = data

    def process(self, x, y):
        return x + y


class OneHot:
    '''
    自定义UDF OneHot
    构造函数，参数labels，为OneHot所需全部label
    '''

    def __init__(self, labels=None):
        self.labels = labels

    def process(self, x):
        '''
        udf OneHot 处理
        :param x: 输入数据
        :return: list, 按照self.data 中全部label编码输出
        '''
        result = [1 if label == x else 0 for label in self.labels]
        is_other = 0 if x in self.labels else 1
        result.append(is_other)
        return result



class MovingWindow:
    '''
    自定义UDF window
    构造函数，参数data，为Window所需窗口
    '''
    def __init__(self, moving_wide_list=None):
        self.moving_wide_list = moving_wide_list
        self.windows_num = len(self.moving_wide_list)

    def process(self, x):
        result = [0.0 for _ in range(self.windows_num)]
        for index in range(self.windows_num - 1):
            if len(x) < self.moving_wide_list[index + 1] - 1:  # 如果长度不足当前窗口大小
                result[index] = sum(x[self.moving_wide_list[index]:])
            else:  # 如果长度大于当前窗口大小
                result[index] = sum(x[self.moving_wide_list[index]:self.moving_wide_list[index + 1]])
        if len(x) >= self.moving_wide_list[-1]:  # 如果长度达到最后一个窗口
            result[-1] = sum(x[self.moving_wide_list[-1]:])
        return result


class Bucket:
    '''
    自定义UDF bucket
    '''

    def __init__(self, data=None):
        if isinstance(data, list):
            self.data = data
        else:
            logging.error(self.data)
            raise Exception("error bucket, bucket should be a list")

    def process(self, x):
        for index, item in enumerate(self.data):
            if x < item:
                return index
        return len(self.data)


class WeightCategory:
    '''
    自定义UDF bucket
    '''

    def __init__(self, data=None, type=None):
        if isinstance(data, list):
            self.mappings = {value: index for index, value in enumerate(data)}
        else:
            logger.debug(data)
            raise Exception("error bucket, bucket should be a list")

    def process(self, x):
        if isinstance(x, list):
            return [self.mappings[item] for item in x]
        else:
            return self.mappings[x]


class AppList:
    def __init__(self, dict=None):
        '''
        初始化AppList 处理方法
        :param dict: 映射表
        '''

        self.dict = dict
        self.cats = list(set(dict.values())).sort() + ["app_other"]

    def process(self, x):
        '''
        udf Applist 处理
        :param x: 输入数据
        :return: list, 按照self.data 中map进行转换
        '''
        apps = set(x.split("|"))
        cat_map = {}
        for app in apps:
            if '0' == app or '\\N' == app:
                continue
            cat = self.dict.get(app, "app_other")
            cat_map.setdefault(cat, set())
            cat_map[cat].add(app)

        result = []
        for cat in self.cats:
            if cat in cat_map:
                result.append(len(cat_map.get(cat)))
            else:
                result.append(0)
        return result


class Markerinfo:
    def __init__(self, statistics_by_day_columns=None, sum_on_para_by_day=None,
                 one_hot_on_latest_day_feature_columns=None):
        self.statistics_by_day_feature_columns = statistics_by_day_columns
        self.sum_on_para_by_day = sum_on_para_by_day
        self.one_hot_on_latest_day_feature_columns = one_hot_on_latest_day_feature_columns

    def process(self, x):
        marker_info = json.loads(x) if '\\N' != x else {}
        count_map = {}

        for one_k, one_actinfo_list in marker_info.items():
            if one_k in self.statistics_by_day_feature_columns:  # 按天求次数
                count_map[one_k] = len(one_actinfo_list)
            if one_k in self.sum_on_para_by_day:  # 按天累加相应的参数值
                one_para = self.sum_on_para_by_day[one_k]
                kname_para = "{key}-{para}".format(key=one_k, para=one_para)

                for one_act_info in one_actinfo_list:
                    if "ext_data" in one_act_info:
                        if one_para in one_act_info["ext_data"]:
                            try:
                                one_add_v = float(one_act_info["ext_data"][one_para])
                            except Exception as e:
                                one_add_v = 0
                            count_map[kname_para] = one_add_v

            for one_act_info in one_actinfo_list:  # 需要最后一天有效数据处理成onehot的参数值
                for one_para_k, one_para_v in one_act_info.items():
                    if one_para_k in self.one_hot_on_latest_day_feature_columns:
                        count_map[one_para_k] = one_para_v

        result = []
        for one_k in self.statistics_by_day_feature_columns:
            result.append(count_map.get(one_k, 0))
        for one_k in self.sum_on_para_by_day:
            result.append(count_map.get(one_k, 0))
        for one_k in self.one_hot_on_latest_day_feature_columns:
            result.append(count_map.get(one_k, 0))
        return result


class ModelSelect:
    def process(self, x):
        pass
