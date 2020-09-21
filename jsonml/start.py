from jsonml import source
from pandas import DataFrame
import json
from jsonml import dataprocess
from jsonml.dataprocess import MDataFrame
import numpy as np
import importlib
from jsonml import datautil
from jsonml import datesutil
import copy
import time
import jsonml.model as mmodel
from jsonml.model import ModelProcess
import pandas as pd
from sklearn.model_selection import train_test_split
import gc
import logging
import re

import os

logger = logging.getLogger('jsonml')


group_udf_mapping = {
    'Max': 'max',
    'Min': 'min',
    'Mean': 'mean',
    'Sum': 'sum',
    'Count': 'size',
    "Std": "std",
    "Var": "var",
    "Sem": "sem",
    "FirstValid": "first",
    "LatestValid": "last",
    "NthValid": "nth"
}


def load_udf(name, paramlist):
    if "." in name:
        parts = name.split(".")
        mod_name = ".".join([parts[i] for i in range(0, len(parts) - 1)])
        cls_name = parts[-1]
    else:
        mod_name = 'jsonml.udf'
        cls_name = name
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    udf = cls(*paramlist)
    return udf


class Pipeline:
    def __init__(self, config_path=None, **xxkwargs):
        root_path = os.path.dirname(os.path.abspath(__file__))
        self.udf_param_dict = source.read_json(os.path.join(root_path, 'udf_param.json'))
        logging.debug(self.udf_param_dict)

        self.config = source.read_json(config_path)
        level = logging.DEBUG if 'debug' in self.config and self.config['debug'] else logging.INFO
        logger.setLevel(level)
        self.config = self.check_config(self.config, **xxkwargs)

        self.shuffles = self.parse_process_data_stage()
        logging.debug(self.shuffles)
        self.shuffle_data = []
        self.current_batch_index = 0

        self.is_predict = True if 'model' in self.config and "run_mode" in self.config['model'] and \
                                   self.config['model']['run_mode'] == 'predict' else False

    def batch_process(self, df):
        '''
        流式处理一批数据，
        当前为无序数组时，会执行第一个shuffle之前的操作
        当前为有序数据时，会处理完所有数据
        当前为预测模式时，会处理完所有数据，并且进行模型处理和输出
        :param df: 输入df
        :return: None
        '''
        self.current_batch_index = self.current_batch_index + 1
        if len(self.shuffles) > 0:
            for stage in self.shuffles[0]:
                df = self.process_stage(df, stage)
        self.shuffle_data.append(df)

        if self.is_predict:
            df = self.shuffle_process()
            self.process_model(df)

    def shuffle_process(self):
        '''
        流式处理一批数据，
        当前为无序数组时，会执行第一个shuffle之后的操作
        当前为有序数据时，会进行所有的数据合并处理
        当前为预测模式时，不进行任何处理
        :return: 处理后的数据
        '''
        gc.collect()
        if len(self.shuffle_data) == 0:
            return None
        df = self.shuffle_data[0] if len(self.shuffle_data) == 0 else pd.concat(self.shuffle_data, axis=0, ignore_index=True, sort=False)
        self.shuffle_data = []

        for i in range(1, len(self.shuffles)):
            for stage in self.shuffles[i]:
                df = self.process_stage(df, stage)
        return df

    def check_config(self, config, **xxkwargs):
        '''
        配置替换
        将原来所有的文件写法的配置，还原成全量配置；去掉注释
        :param config: 老的配置
        :param xxkwargs: 需要替换的变量
        :return: 替换后的配置
        '''
        new_config = {}
        for key, value in config.items():
            if "notes" == key:
                continue

            if key.endswith("_file") and isinstance(value, str):
                if "[" in value:
                    parts = value.split("[")
                    key_config = source.read_json(parts[0])
                    for index in range(1, len(parts)):
                        part = parts[index][:-1]
                        sub_key = part if datautil.str2int(part, -1) == -1 else datautil.str2int(part, -1)
                        key_config = key_config[sub_key]
                else:
                    key_config = source.read_json(value)
                new_config[key[:-5]] = key_config
            elif isinstance(value, str) and '$' in value:
                params = re.findall(r'\${.*}', value)
                for param in params:
                    param = param[2:-1]
                    if param in xxkwargs:
                        new_config[key] = re.sub('\${.*}', xxkwargs[param], value)
            elif isinstance(value, dict):
                new_config[key] = self.check_config(value, **xxkwargs)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                new_config[key] = [self.check_config(item, **xxkwargs) for item in value]
            else:
                new_config[key] = value
        return new_config

    def parse_params(self, strategy, name):
        '''
        解析一个udf里面所有的param参数
        :param strategy: udf策略字段
        :param name: udf名称
        :return: param参数list
        '''
        params = []
        for key, value in strategy.items():
            key = key if name not in self.udf_param_dict or key not in self.udf_param_dict[name] else self.udf_param_dict[name][key]
            if "param" in key:
                params.append((int(key[5:]), value))
        params.sort()
        return [value for (key, value) in params]

    def read_data(self):
        '''
        数据读入
        :return: df
        '''
        csource = self.config['source']
        type = "text" if "type" not in csource else csource['type']
        cinput = csource['input']
        if type == "text":
            columns = None if "columns" not in csource else csource['columns']
            data_type = 'str' if "data_type" not in csource else csource['data_type']
            select_columns = [] if "select_columns" not in csource else csource["select_columns"]
            drop_columns = [] if "drop_columns" not in csource else csource["drop_columns"]
            key_columns = [] if "key_columns" not in csource else csource["key_columns"]
            keep_key_columns = True if "keep_key_columns" not in csource else csource["keep_key_columns"]
            filter = '' if "filter" not in csource else csource["filter"]

            column_info = []
            if columns is not None:
                for index, column in enumerate(columns):
                    parts = column.strip().split(":") if len(column.strip()) > 0 else ["value-" + str(index + 1)]
                    if len(parts) == 1:
                        column_info.append([parts[0], 'str', ''])
                    elif len(parts) == 2:
                        column_info.append([parts[0], parts[1], ''])
                    elif len(parts) == 3:
                        column_info.append([parts[0], parts[1], parts[2]])
                    elif len(parts) > 3:
                        column_info.append([parts[0], parts[1], ':'.join(parts[2:])])
                    else:
                        raise Exception(column)


            path = cinput["path"]
            is_stream = False if "is_stream" not in cinput else cinput['is_stream']
            if not is_stream:
                args = ['field_delimiter', 'ignore_first_line', 'ignore_error_line', 'ignore_blank_line']
                kwargs = {key: cinput[key] for key in args if key in cinput}
                df = source.csv(path, columns=column_info, **kwargs)
                if len(select_columns) > 0:
                    for column in drop_columns:
                        if column in select_columns:
                            select_columns.remove(column)
                    df = df[[select for select in select_columns]]
                elif len(drop_columns) > 0:
                    df.drop(drop_columns, axis=1, inplace=True)

                if len(key_columns) > 0:
                    if not keep_key_columns:
                        names = {key: "keys_" + key for key in key_columns}
                        df.rename(columns=names, inplace=True)
                    else:
                        for key in key_columns:
                            df["keys_" + key] = df[key]

                if isinstance(filter, str) and filter != '':
                    df.query(filter, inplace=True)
                    df.reset_index(drop=True, inplace=True)
                elif isinstance(filter, dict):
                    name = filter["name"]
                    filter_columns = filter["input_columns"]
                    udf = load_udf(name, self.parse_params(filter, name))
                    mask = df.apply(lambda row: udf.process(*tuple([row[filter_column] for filter_column in filter_columns])),axis=1)
                    df = df[mask]
                    df.reset_index(drop=True, inplace=True)
                if data_type in ['int', 'float']:
                    df = df.apply(pd.to_numeric)
                return df
            else:
                def callback(df):
                    if len(select_columns) > 0:
                        for column in drop_columns:
                            if column in select_columns:
                                select_columns.remove(column)
                        df = df[[select for select in select_columns]]
                    if len(drop_columns) > 0:
                        df.drop(drop_columns, axis=1, inplace=True)

                    if len(key_columns) > 0:
                        if not keep_key_columns:
                            names = {key: "keys_" + key for key in key_columns}
                            df.rename(columns=names, inplace=True)
                        else:
                            for key in key_columns:
                                df["keys_" + key] = df[key]

                    if isinstance(filter, str) and filter != '':
                        df.query(filter, inplace=True)
                        df.reset_index(drop=True, inplace=True)
                    elif isinstance(filter, dict):
                        name = filter["name"]
                        filter_columns = filter["input_columns"]
                        udf = load_udf(name, self.parse_params(filter, name))
                        mask = df.apply(lambda row: udf.process(*tuple([row[filter_column] for filter_column in filter_columns])), axis=1)
                        df = df[mask]
                        df.reset_index(drop=True, inplace=True)
                    if data_type in ['int', 'float']:
                        df = df.apply(pd.to_numeric)
                    self.batch_process(df)

                args = ['field_delimiter', 'ignore_first_line', 'ignore_error_line', 'ignore_blank_line', 'batch_count', 'batch_key']
                kwargs = {key: cinput[key] for key in args if key in cinput}

                if path == "stdin":
                    source.stdin_stream(columns=column_info, callback=callback, **kwargs)
                else:
                    source.csv_stream(path, columns=column_info, callback=callback, **kwargs)
        elif type == "es":
            logger.error("do not support now")
        else:
            logger.error("do not support now")

    def process_data(self, df):
        '''
        非流式数据处理
        :param df: 输入数据
        :return: 处理后的数据
        '''
        if 'process' not in self.config:
            return df
        config = self.config['process']
        stages = [(stage_id, stage) for stage_id, stage in config.items()]
        stages.sort(key=lambda elem: int(elem[0].split("_")[1]))
        for stage_id, stage in stages:
            df = self.process_stage(df, stage)
        return df

    def process_stage(self, df, stage):
        '''
        一个stage的数据处理，流式和非流式均如此
        :param df: 输入数据
        :param stage: stage详细配置
        :return: 执行后的数据
        '''
        start = time.time()
        stage_type = 'map' if 'type' not in stage else stage['type']
        strategies = stage['strategies']
        if stage_type == 'group':
            group_keys = stage['group_key_columns']
            sort_keys = [] if 'sort_key_columns' not in stage else stage['sort_key_columns']
            keep_group_keys = True if 'keep_group_keys' not in stage else stage['keep_group_keys']
            df = self.group_stage(df, strategies, group_keys, sort_keys, keep_group_keys)
            logger.info('********* end group stage: group keys = ' + str(group_keys) +
                         ' cost = ' + str(time.time() - start) + ' **********')
        elif stage_type == 'map':
            mdf = MDataFrame(df)
            for strategy in strategies:
                self.process_strategy(mdf, strategy)
            df = mdf.datas()
            logger.info('********* end map stage: cost = ' + str(time.time() - start) + ' **********')
        logger.debug(df)
        logger.debug(df.columns.values.tolist())

        return df

    def parse_process_data_stage(self):
        '''
        把stages分成若干shuffle，用于流式非有序数据执行
        :return: 分开后的shuffle list配置
        '''
        shuffles = []
        if 'process' not in self.config:
            return shuffles

        is_sorted = True if 'is_sorted' not in self.config['source'] else self.config['source']['is_sorted']
        config = self.config['process']

        stages = [(stage_id, stage) for stage_id, stage in config.items()]
        stages.sort(key=lambda elem: int(elem[0].split("_")[1]))

        shuffle_stages = []
        for stage_id, stage in stages:
            stage_type = 'map' if 'type' not in stage else stage['type']
            if stage_type == 'group':
                shuffle_stages.append(stage)
                if not is_sorted:
                    shuffles.append(shuffle_stages)
                    shuffle_stages = [stage]
            elif stage_type == 'map':
                shuffle_stages.append(stage)
        if len(shuffle_stages) > 0:
            shuffles.append(shuffle_stages)
        return shuffles

    def group_stage(self, df, strategies, group_keys, sort_keys, keep_group_keys):
        '''
        group stage数据处理
        :param df: 输入数据
        :param strategies: group策略
        :param group_keys: group的key
        :param sort_keys:  group后内部排序的key
        :param keep_group_keys: 是否保留group key
        :return: group stage数据处理后的数据
        '''
        df_columns = df.columns.values.tolist()
        strategies_list = self.parse_group_params(strategies, df_columns, group_keys)
        logger.debug(strategies_list)

        if sort_keys is not None and len(sort_keys) > 0:
            df.sort_values(sort_keys, inplace=True)
        result_df = df.groupby(by=group_keys, as_index=False, sort=False).agg(strategies_list)

        mdf = MDataFrame(result_df)
        udf = load_udf('Copy', [])

        output_columns = ["keys_" + column for column in group_keys]
        mdf.process_udf(udf, copy.deepcopy(group_keys), output_columns, keep_group_keys)
        return mdf.datas()

    def parse_group_params(self, strategies, src_columns, keys):
        '''
        解析group的配置
        :param strategies:
        :param src_columns:
        :param keys:
        :return:
        '''
        processed_columns = copy.deepcopy(keys)
        strategies_dict = {}
        for strategy in strategies:
            logger.debug(strategy)
            if "input_columns" in strategy:
                input_columns = strategy["input_columns"]
                processed_columns.extend(input_columns)
            elif "default" in strategy and strategy['default']:
                input_columns = copy.deepcopy(src_columns)
                for column in src_columns:
                    if column in processed_columns:
                        input_columns.remove(column)
            else:
                raise Exception("input_columns can not be empty!")

            logger.debug(input_columns)

            name = strategy["name"]
            for input_column in input_columns:
                if name in group_udf_mapping:
                    strategies_dict[input_column] = group_udf_mapping[name]
                else:
                    strategies_dict[input_column] = load_udf(name, self.parse_params(strategy, name)).process
        return strategies_dict

    def process_strategy(self, mdf, strategy, strategy_names=None):
        '''
        udf处理
        :param mdf: 输入数据
        :param strategy: udf配置
        :param strategy_names: udf名字，udf支持name在外和在内两种
        :return: 处理后的数据
        '''
        names = strategy_names if strategy_names is not None else strategy["name"]
        if names == "Output":
            self.save_file(mdf.data, strategy)
            return None
        if names == "GroupAuc":
            df = mdf.datas()
            key_columns = strategy["key_columns"]
            key_data = df[key_columns[0]].tolist() if len(key_columns) == 1 else [tuple(x) for x in df[key_columns].values]
            group_auc, detail_auc = mmodel.cal_group_auc(df['label'].tolist(), df['pred_prob'].tolist(), key_data)
            logger.info(f'group_auc = {group_auc}')
            if strategy["detail"]:
                logger.info(f'detail_auc : ')
                for key, auc in detail_auc.items():
                    logger.info(f'key = {key}, auc = {auc}')
            return None
        elif names == "RenameColumn":
            input_columns = strategy["input_columns"]
            output_columns = strategy["output_columns"]
            columns_dict = {}
            for index, input in enumerate(input_columns):
                columns_dict[input] = output_columns[index]
            mdf.rename(columns_dict)
            return None
        elif names == "CopyColumn":
            input_columns = strategy["input_columns"]
            output_columns = strategy["output_columns"]
            mdf.copy_column(input_columns, output_columns)
            return None
        elif names == "AddColumn":
            input_columns = strategy["input_columns"]
            value = strategy['value']
            mdf.add_column(input_columns, value)
            return None
        elif names == "DropColumn":
            mdf.drop(strategy["input_columns"])
            return None
        elif names == "OrderColumn":
            columns = strategy["input_columns"]
            if isinstance(columns, str) and "," in columns:
                columns = columns.split(",")
            columns = [column.strip() for column in columns]
            # 增加key columns，放在最前面
            key_column = [column for column in mdf.columns() if column.startswith('keys_') and column not in columns]
            if len(key_column) > 0:
                key_column.extend(columns)
                columns = key_column
            mdf.order_column(columns)
            return None

        input_columns = copy.deepcopy(strategy["input_columns"])
        if isinstance(input_columns, dict):
            logger.debug("****** parse sub strategy *******")
            input_columns = self.process_strategy(mdf, input_columns)

        output_columns = copy.deepcopy(input_columns) if "output_columns" not in strategy else copy.deepcopy(strategy[
            "output_columns"])

        split_column_count = 0 if "split_column_count" not in strategy else strategy["split_column_count"]
        suffix_use_label = False if "suffix_use_label" not in strategy else strategy["suffix_use_label"]
        if suffix_use_label and "labels" in strategy:
            labels = copy.deepcopy(strategy["labels"])
            default_label = 'others' if 'default_label' not in strategy else strategy['default_label']
            labels.append(default_label)
            for index, output_column in enumerate(output_columns):
                pre = output_column if not isinstance(output_column, list) else output_column[0]
                output_columns[index] = [pre + '_' + str(label) for label in labels]
        elif split_column_count > 1:
            for index, output_column in enumerate(output_columns):
                pre = output_column if not isinstance(output_column, list) else output_column[0]
                output_columns[index] = [pre + '_' + str(i) for i in range(split_column_count)]

        prefix = "" if "output_columns_prefix" not in strategy else strategy["output_columns_prefix"]
        suffix = "" if "output_columns_suffix" not in strategy else strategy["output_columns_suffix"]
        for index, output_column in enumerate(output_columns):
            output_columns[index] = prefix + output_column + suffix if not isinstance(output_column, list) \
                else [prefix + column + suffix for column in output_column]

        keep_input_columns = False if "keep_input_columns" not in strategy else strategy["keep_input_columns"]

        names = names if isinstance(names, list) else [names]

        logger.debug("*********  start to execute strategy " + str(names) + " **********")
        logger.debug("input_columns: " + str(input_columns))
        logger.debug("output_columns: " + str(output_columns))
        start = time.time()

        for name in names:
            udf = load_udf(name, self.parse_params(strategy, name))
            mdf.process_udf(udf, input_columns, output_columns, keep_input_columns)
        if "drop_columns" in strategy:
            mdf.drop(strategy["drop_columns"])
        if "select_columns" in strategy:
            mdf.select(strategy["select_columns"])
        logger.debug(mdf)
        logger.debug(mdf.columns())
        cost = time.time() - start
        logger.debug("*********  stop to execute strategy " + str(names) + " cost = " + str(cost) + " **********")
        return output_columns

    def process_model(self, df):
        '''
        模型处理
        :param df: 输入输出
        :return: 无
        '''

        if 'model' not in self.config:
            logger.info("no model in json, ignore model process")
            return
        gc.collect()
        config = self.config['model']
        columns = df.columns.values.tolist()

        logger.info("********* start process mode ********")
        logger.debug(columns)
        logger.debug(df)
        run_mod = 'train_test' if "run_mode" not in config else config["run_mode"]

        models = []
        for key, model in config.items():
            if key.startswith("model_"):
                models.append((key[6:], model))
        models.sort()
        logger.debug(models)

        if run_mod == "predict":
            model_select = 'ModelSelect' if "model_select" not in config else config["model_select"]
            group_keys = [column for column in columns if column.startswith('keys_')]
            group_key_df = df[group_keys]
            df.drop(group_keys, axis=1, inplace=True)

            for _, model in models:
                logger.debug(model)
                if "model_path" not in model:
                    raise Exception("model_path could not be null!")
                model_path = model["model_path"]
                model_process = ModelProcess()
                model_process.load_model(model_path=model_path)
                pred_df = model_process.predict(df)

                group_key_df.columns = [key[5:] for key in group_keys]
                df_temp = pd.concat([group_key_df, pred_df], axis=1, sort=False)

                if 'strategies' in model:
                    strategies = model['strategies']
                    mdf = MDataFrame(df_temp)
                    for strategy in strategies:
                        self.process_strategy(mdf, strategy)
                    df_temp = mdf.datas()
                if 'Output' in model:
                    self.save_file(df_temp, model['Output'])
        elif run_mod == "train":
            group_keys = [column for column in columns if column.startswith('keys_')]
            df.drop(group_keys, axis=1, inplace=True)

            validation_data_percent = 0.2 if "validation_data_percent" not in config else config[
                "validation_data_percent"]
            validation_data_percent = 0.2 if validation_data_percent > 0.5 or validation_data_percent < 0.01 else validation_data_percent

            x_df, y_df = dataprocess.split_feature_and_label_df(df)
            del df
            train_x_df, valid_x_df, train_y_df, valid_y_df = train_test_split(x_df, y_df,
                                                                              test_size=validation_data_percent, random_state=0)
            del x_df, y_df
            for _, model in models:
                logger.debug(model)
                model_type = model["model_type"]
                model_config = model["model_config"]

                model_process = ModelProcess(model_type, model_config)
                model_process.train_model(train_x_df, train_y_df, test_x=valid_x_df, test_y=valid_y_df)
                model_process.save_model(model["model_path"])
                logger.info("model saved to " + os.path.abspath(model["model_path"]))

                if 'feature_importance' in model:
                    feature_importance = model['feature_importance']
                    importance_types = ['gain'] if 'importance_type' not in feature_importance else feature_importance['importance_type']

                    for importance_type in importance_types:
                        score = model_process.feature_importance(importance_type)

                        all_features = [score.get(f, 0.) for f in model_process.features()]
                        all_features = np.array(all_features, dtype=np.float32)
                        all_features_sum = all_features.sum()

                        importance_list = [[f, score.get(f, 0.) / all_features_sum] for f in model_process.features()]
                        importance_list.sort(key=lambda elem: elem[1], reverse=True)

                        print("feature importance: " + importance_type)
                        for index, item in enumerate(importance_list):
                            print(index, item[0], item[1])

        elif run_mod == "test":
            group_keys = [column for column in columns if column.startswith('keys_')]
            group_key_df = df[group_keys]
            df.drop(group_keys, axis=1, inplace=True)
            x_df, y_df = dataprocess.split_feature_and_label_df(df)
            del df
            for _, model in models:
                logger.debug(model)
                if "model_path" not in model:
                    raise Exception("model_path could not be null!")
                model_process = ModelProcess()
                model_process.load_model(model_path=model["model_path"])
                pred_df = model_process.evaluate_model(x_df, y_df, ana_top=0.05)

                group_key_df.columns = [key[5:] for key in group_keys]
                df_temp = pd.concat([group_key_df, y_df, pred_df], axis=1, sort=False)

                if 'strategies' in model:
                    strategies = model['strategies']
                    mdf = MDataFrame(df_temp)
                    for strategy in strategies:
                        self.process_strategy(mdf, strategy)
                    df_temp = mdf.datas()
                if 'Output' in model:
                    self.save_file(df_temp, model['Output'])

        elif run_mod == "train_test":
            group_keys = [column for column in columns if column.startswith('keys_')]
            group_key_df = df[group_keys]
            df.drop(group_keys, axis=1, inplace=True)

            test_data_percent = 0.2 if "test_data_percent" not in config else config["test_data_percent"]
            test_data_percent = 0.2 if test_data_percent > 0.5 or test_data_percent < 0.01 else test_data_percent

            validation_data_percent = 0.2 if "validation_data_percent" not in config else config["validation_data_percent"]
            validation_data_percent = 0.2 if validation_data_percent > 0.5 or validation_data_percent < 0.01 else validation_data_percent

            x_df, y_df = dataprocess.split_feature_and_label_df(df)
            del df
            train_x_df, test_x_df, train_y_df, test_y_df = train_test_split(x_df, y_df, test_size=test_data_percent, random_state=0)
            del x_df, y_df
            train_x_df, valid_x_df, train_y_df, valid_y_df = train_test_split(train_x_df, train_y_df, test_size=validation_data_percent, random_state=0)

            for _, model in models:
                logger.debug(model)
                model_process = ModelProcess(model["model_type"], model["model_config"])
                model_process.train_model(train_x_df, train_y_df, test_x=valid_x_df, test_y=valid_y_df)
                model_process.save_model(model["model_path"])
                logger.info("model saved to " + os.path.abspath(model["model_path"]))

                pred_df = model_process.evaluate_model(test_x_df, test_y_df, ana_top=0.05)
                group_key_df.columns = [key[5:] for key in group_keys]
                df_temp = pd.concat([group_key_df, test_y_df, pred_df], axis=1, sort=False)

                if 'strategies' in model:
                    strategies = model['strategies']
                    mdf = MDataFrame(df_temp)
                    for strategy in strategies:
                        self.process_strategy(mdf, strategy)
                    df_temp = mdf.datas()
                if 'Output' in model:
                    self.save_file(df_temp, model['Output'])


    def save_file(self, src_df, strategy):
        '''
        文件保存，或结果输出
        :param src_df: 数据
        :param strategy: 输出策略
        :return: 无
        '''
        df = src_df.copy(deep=True)

        columns = df.columns.values.tolist()
        key_columns = [column for column in columns if column.startswith('keys_')]
        if len(key_columns) > 0:
            group_keys = {column:column[5:] for column in key_columns}
            df.drop([column[5:] for column in key_columns if column[5:] in columns], axis=1, inplace=True)
            df.rename(columns=group_keys, inplace=True)

        path = 'pipeline.txt' if 'path' not in strategy else strategy['path']
        type = 'text' if 'type' not in strategy else strategy['type']
        if path == "stdout":
            field_delimiter = ',' if 'field_delimiter' not in strategy else strategy['field_delimiter']
            columns = None if 'columns' not in strategy else strategy['columns']
            if columns:
                df = df[[column for column in columns]]
            source.stdout(df, field_delimiter)
        elif type == 'text':
            field_delimiter = ',' if 'field_delimiter' not in strategy else strategy['field_delimiter']
            columns = None if 'columns' not in strategy else strategy['columns']
            header = False if self.current_batch_index > 1 else True if 'header' not in strategy else strategy['header']
            path = path if not path.endswith("/") else path + time.strftime("%Y%m%d%H%M%S", time.localtime()) + ".txt"
            filepath, _ = os.path.split(path)
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            df.to_csv(path, sep=field_delimiter, columns=columns, header=header, mode='a+')
        elif type == "excel":
            df.to_excel()
        else:
            logger.info("we will support type " + type + " later")


def start(config_path, **xxkwargs):
    start0 = time.time()

    pipeline = Pipeline(config_path, **xxkwargs)
    logger.info("read and parse config cost = " + str(time.time() - start0))

    start1 = time.time()
    df = pipeline.read_data()

    if df is not None:
        logger.info("read data cost = " + str(time.time() - start1))
        logger.debug(df)
        start1 = time.time()
        df = pipeline.process_data(df)
        logger.info("process data cost = " + str(time.time() - start1))
    else:
        start1 = time.time()
        df = pipeline.shuffle_process()
        logger.info("process data cost = " + str(time.time() - start1))

    if df is not None:
        start1 = time.time()
        pipeline.process_model(df)
        logger.info("process model cost = " + str(time.time() - start1))

    logger.info("all cost = " + str(time.time() - start0))


if __name__ == "__main__":
    start("common_config.json")

