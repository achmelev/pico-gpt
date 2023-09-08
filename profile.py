from time import time, strftime, gmtime

timers = {}
format_string = "%H:%M:%S"

def create_timer(name):
    global timers
    assert name not in timers, "timer "+name+" does already exist"
    m = {"sum": 0.0, "count" : 0, "time": None}
    timers[name] = m

def start(name):
    global timers
    assert name in timers, "timer "+name+", doesn't exist"
    assert timers[name]["time"] == None, "timer "+name+", already started"
    ct = time()
    timers[name]["time"] = ct

def stop(name):
    global timers
    assert name in timers, "timer "+name+", doesn't exist"
    assert timers[name]["time"] != None, "timer "+name+", wasn't started"
    ct = time()
    st = timers[name]["time"]
    timers[name]["time"] = None
    timers[name]["count"] = 1 + timers[name]["count"]
    timers[name]["sum"] = (ct - st) + timers[name]["sum"]

def fmt_time(seconds):
    global format_string
    return strftime(format_string, gmtime(seconds))


def get_time_sum(name):
    global timers
    assert name in timers, "timer "+name+", doesn't exist"
    return timers[name]["sum"]

def get_time_sum_fmt(name):
    return (fmt_time(get_time_sum(name)))

def get_count(name):
    global timers
    assert name in timers, "timer "+name+", doesn't exist"
    return timers[name]["count"]

def get_time_avg(name):
    global timers
    assert name in timers, "timer "+name+", doesn't exist"
    if timers[name]["count"] == 0:
        return 0.0
    else:
        return timers[name]["sum"]/timers[name]["count"]

def get_time_avg_fmt(name):
    return (fmt_time(get_time_avg(name)))








