import time
import datetime


def today(format='%Y%m%d'):
    '''
    今天日期
    :param format: 日期格式
    :return: string '20200101'
    '''
    return datetime.date.today().strftime(format)


def yesterday(format='%Y%m%d'):
    '''
    昨天日期
    :param format: 日期格式
    :return: string '20200101'
    '''
    return (datetime.date.today() - datetime.timedelta(days=1)).strftime(format)


def tomorrow(format='%Y%m%d'):
    '''
    明天日期
    :param format: 日期格式
    :return: string '20200101'
    '''
    return (datetime.date.today() + datetime.timedelta(days=1)).strftime(format)


def future_day(days, format='%Y%m%d'):
    '''
    未来几天的日期
    :param days: int 未来与今天天数间隔
    :param format: 日期格式
    :return: string '20200101'
    '''
    return (datetime.date.today() + datetime.timedelta(days=days)).strftime(format)


def past_day(days, format='%Y%m%d'):
    '''
    过去几天的日期
    :param days: int 过去与今天天数间隔
    :param format: str 日期格式
    :return: str '20200101'
    '''
    return (datetime.date.today() - datetime.timedelta(days=days)).strftime(format)


def timestamp_to_datetime(timestamp, format="%Y-%m-%d %H:%M:%S"):
    '''
    时间戳转化为日期字符串
    %Y-年（四位数）; %m-月; %d-日(月中的一天); %H-时; %M:分; %S-秒;
    :param timestamp: int  时间戳
    :param format: str  日期格式
    :return: str  日期
    '''
    timestamp = int(timestamp)
    timeArray = time.localtime(timestamp)
    datetime = time.strftime(format, timeArray)
    return datetime


def datetime_to_timestamp(datetime, format="%Y-%m-%d %H:%M:%S"):
    '''
    日期字符串转化为时间戳
    %Y-年（四位数）; %m-月; %d-日(月中的一天); %H-时; %M:分; %S-秒;
    :param datetime: str  日期
    :param format:
    :return: int timestamp 秒
    '''
    timestamp = int(time.mktime(time.strptime(datetime, format)))
    return timestamp


def get_delta_days_by_timestamp(t1, t2):
    '''
    根据时间戳t1和t2，计算整两个时间戳所对应的日期的天数之差。dt1-dt2
    :param t1: int  时间戳
    :param t2: int  时间戳
    :return: delta_days = int t1和t2所在日期的天数之差
    '''
    dt1 = datetime.datetime.fromtimestamp(t1)
    dt1 = dt1.replace(hour=0, minute=0, second=0, microsecond=0)
    dt2 = datetime.datetime.fromtimestamp(t2)
    dt2 = dt2.replace(hour=0, minute=0, second=0, microsecond=0)

    return (dt1 - dt2).days


def get_delta_days(t1, t2, format="%Y%m%d"):
    '''
    根据时间戳t1和t2，计算整两个时间戳所对应的日期的天数之差。dt1-dt2
    :param t1: str  时间戳
    :param t2: int  时间戳
    :return: delta_days = int t1和t2所在日期的天数之差
    '''
    dt1 = datetime.datetime.strptime(t1, format)
    dt1 = dt1.replace(hour=0, minute=0, second=0, microsecond=0)
    dt2 = datetime.datetime.strptime(t2, format)
    dt2 = dt2.replace(hour=0, minute=0, second=0, microsecond=0)

    return (dt1 - dt2).days


def get_nday_list(start, n, reverse=True):
    start = datetime.datetime.strptime(start,"%Y%m%d")
    before_n_days = []
    for i in range(0, n):
        gap = - datetime.timedelta(days=i) if reverse else datetime.timedelta(days=i)
        before_n_days.append((start + gap).strftime("%Y%m%d"))
    return before_n_days
