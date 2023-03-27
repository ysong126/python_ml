import datetime
import pandas as pd
import numpy as np

def example1():
    now = datetime.datetime.now()
    print(f'current time is: {now}')
    return

def example2():
    time = datetime.datetime(2020,11,5)
    print(f'created time is :{time}')
    return

def example3():
    time0 = datetime.datetime(2019,12,18,15)
    print(f'using strftime to convert {time0} to {time0.strftime("%b")}')
    print(f'That day is a {time0.strftime("%A. %b %d, %Y")}')
    return

# pandas examples

def example4():
    dti = pd.to_datetime(
        ["1/1/2018",np.datetime64("2018-01-01"),datetime.datetime(2018,1,1)]
    )
    print(dti)
    return

def example5():
    dti = pd.date_range("2018-01-01",periods=5, freq="D")
    print(f'Generated sequence of date is {dti}')
    return

def example6():
    dti = pd.date_range("2018-01-01", periods=5, freq="D")
    dti = dti.tz_localize("UTC")
    print(f'current time is {dti}')

    new_zone_dti = dti.tz_convert("US/PAcific")
    print(f'converted date is {new_zone_dti}')
    return

def example7():
    day = pd.Timestamp("2021-06-03")
    print(f'day is {day}')
    day2 = day+pd.Timedelta("2 day")
    print(f'day2 is {day2}')
    print(f'day name is {day2.day_name()}')
    print(f'next businees day is {day2+pd.offsets.BDay()}')
    return