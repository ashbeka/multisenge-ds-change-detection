import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional


@dataclass
class TimerData:
    elapsed: float
    start_time: float
    timer_name: Optional[str]
    print_on_exit: bool


@contextmanager
def timer(*, timer_name: Optional[str] = None, print_on_exit: bool = False):
    """タイマーのコンテキストマネージャ

    Args:
        timer_name (Optional[str]): タイマー名（ログ出力用）
        print_on_exit (bool): コンテキストを抜けるときにログを出力するかどうか

    Yields:
        TimerData: タイマーデータ
    """
    start_time = time.time()
    time_data = TimerData(
        elapsed=0.0,
        start_time=start_time,
        timer_name=timer_name,
        print_on_exit=print_on_exit,
    )

    try:
        yield time_data
    finally:
        time_data.elapsed = time.time() - start_time
        if print_on_exit and time_data.timer_name:
            print(f"{time_data.timer_name}: {time_data.elapsed:.3f} seconds")


if __name__ == "__main__":
    with timer() as t:
        time.sleep(1)
    print(t.elapsed)

    with timer(timer_name="test", print_on_exit=True) as t:
        time.sleep(0.2)
    print(t)
