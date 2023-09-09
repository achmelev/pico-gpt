import unittest
from time import sleep
from timers import create_timer, get_count, get_time_avg_fmt, get_time_sum_fmt, start, stop

class TimerTest(unittest.TestCase):

    def test_timer(self):
        create_timer('test')
        for i in range(0,3):
            start('test')
            sleep(1.0)
            stop('test')
        
        fmt_sum = get_time_sum_fmt('test')
        fmt_avg = get_time_avg_fmt('test')

        self.assertEqual(get_count('test'), 3)
        self.assertEqual(fmt_sum, "00:00:03")
        self.assertEqual(fmt_avg, "00:00:01")
            


if __name__ == '__main__':
    unittest.main()