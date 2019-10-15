import functools
import math
import time
from dataclasses import dataclass
import tempfile
import typing
from pathlib import Path

import pandas
from memory_profiler import memory_usage
import numpy
import matplotlib.pyplot as plt


def memory_time_dependence(func: callable,
                           args: tuple = None, kws=None,
                           interval: float = 0.01,
                           include_children: bool = False,
                           ) -> typing.Union[numpy.arange, numpy.arange]:
    """return time_list, memory_list of func(*args, **kws)"""
    if args is None:
        args = ()
    if kws is None:
        kws = {}
    mem_usage = memory_usage((func, args, kws), interval=interval,
                             include_children=include_children)
    return numpy.linspace(0, len(mem_usage) * interval,
                          len(mem_usage)), numpy.array(mem_usage)


def time_memory_of_execution(func, args=None, kws=None, interval: float = 0.01,
                             include_children: bool = False,
                             ) -> typing.Tuple[float, float]:
    """return time of memory usage and max memory value"""
    t, mem = memory_time_dependence(func, args, kws, interval,
                                    include_children)
    return t[-1], max(mem)


def timeit_memit(func, args=None, kws=None, interval: float = 0.01,
                 include_children: bool = False, nums_repeats=4,
                 ) -> typing.Tuple[float, float]:
    """return min time and max memory of time_memory_of_execution for
    num_repeats execution
    """
    tcc = float(str("inf"))
    mcc = -float(str("inf"))
    for _ in range(nums_repeats):
        t, m = time_memory_of_execution(func, args, kws, interval,
                                        include_children)
        tcc = min(t, tcc)
        mcc = max(m, mcc)
    return tcc, mcc


def args_kws_memtime_resol(func, args_list: typing.List[tuple] = None,
                           kws_list: typing.List[dict] = None,
                           interval: float = 0.01,
                           include_children: bool = False, mode="usage",
                           ) -> typing.Tuple[list, list]:
    """return list of times and list of memories info
        mode="usage" return list of min t, min mem values
        mode="time_resolve" return list of numpy.arrange of times
        and correspond memory_values
     """
    args_list = args_list or [()]
    kws_list = kws_list or [{}] * len(args_list)
    time_list = []
    mem_list = []
    mode_func = {"usage": time_memory_of_execution,
                 "time_resolve": memory_time_dependence}[mode]
    for idx, (args, kws) in enumerate(zip(args_list, kws_list)):
        time_, memory = mode_func(func, args, kws, interval,
                                  include_children)
        time_list.append(time_)
        mem_list.append(memory)
    return time_list, mem_list


@dataclass
class ProfileVisualiser:
    time_label: str = "t, sec"
    memory_label: str = "memory, MB"
    interval: float = 0.001
    include_children: bool = False
    show: bool = True
    style = "ggplot"
    grid = True
    _res = None
    _args_kws = None

    def get_time_memory_usage(self, func, args=None, kws=None):
        return time_memory_of_execution(func, args, kws,
                                        interval=self.interval,
                                        include_children=self.include_children, )

    def get_timeit_memit(self, func, args=None, kws=None):
        return timeit_memit(func, args, kws, self.interval,
                            self.include_children)

    def get_memory_time_dependence(self, func, args=None, kws=None):
        return memory_time_dependence(func, args, kws, self.interval,
                                      self.include_children)

    def get_args_kws_dependence(self, func, args_list, kws_list=None):
        kws_list = kws_list or [{}] * len(args_list)
        return zip(*args_kws_memtime_resol(func, args_list, kws_list,
                                           self.interval,
                                           self.include_children,
                                           mode="time_resolve"))

    def plot_args_resolution(self, func, args_list=None, kws_list=None,
                             label_list=None,
                             title=None, add_score=True):
        kws_list = kws_list or [{}] * len(args_list)
        label_list = label_list or [f"args={args}\nkws={kws}" for args, kws in
                                    zip(args_list, kws_list)]
        title = title or f"{func.__name__} args_kws resolution"
        t_usage = []
        m_usage = []

        plt.style.use(self.style)
        if add_score:
            fig, (ax_1, ax_2) = plt.subplots(nrows=2, ncols=1)
        else:
            fig, ax_1 = plt.subplots(nrows=1, ncols=1)
            ax_2 = None

        # plot memory/time dependence for each arg in arg_list

        for idx, (t, mem) in enumerate(
                self.get_args_kws_dependence(func, args_list, kws_list)
        ):
            ax_1.plot(t, mem, label=label_list[idx], linestyle="dotted")
            t_usage.append(t[-1])
            m_usage.append(max(mem))
        ax_1.legend()
        ax_1.set_xlabel(self.time_label)
        ax_1.set_ylabel(self.memory_label)
        ax_1.set_title(title)

        # plot scatter max(memory)/min(time)
        if add_score:
            ax_2.scatter(t_usage, m_usage)
            for x, y, txt in zip(t_usage, m_usage, label_list):
                ax_2.annotate(txt, (x, y))
            ax_2.set_xlabel(self.time_label)
            ax_2.set_ylabel(self.memory_label)
        plt.tight_layout()

        if self.grid:
            plt.grid()
        if self.show:
            plt.show()

    def plot_memory_time_resolution(self, func, args=None, kws=None,
                                    label=None, title=None, annotations=None):

        time_list, memory_list = memory_time_dependence(
            func, args, kws,
            interval=self.interval,
            include_children=self.include_children)

        label = label or f"{func.__name__}\nargs={args}\nkws={kws}"
        title = title or f"{func.__name__} memory-time resolution"

        plt.style.use(self.style)
        plt.plot(time_list, memory_list, label=label, linestyle="dotted")
        plt.legend()
        plt.xlabel(self.time_label)
        plt.ylabel(self.memory_label)
        plt.title(title)
        if annotations:
            for x, y, txt in zip(time_list, memory_list, annotations):
                plt.annotate(txt, (x, y))

        if self.show:
            plt.show()


if __name__ == "__main__":

    def goo_n(n: int = 10, *args, **kwargs):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = numpy.random.random((n,n))
            pandas.DataFrame(
                data=data
            ).to_csv(Path(tmpdir)/"file.csv")
        return data


    def func_example(*args, **kwargs):
        goo_n(*args, **kwargs)


    print(
        f"time_memory_of_execution {time_memory_of_execution(func_example, args=(500,))}"
        )

    # print(
    #     f"time_memory_of_execution {memory_time_dependence(func_example, args=(500,))}"
    # )


    visualiser = ProfileVisualiser()
    # visualiser.plot_memory_time_resolution(func_example, args=(2000, ))
    args_list = [
        (i, ) for i in range(500, 2500, 500)
    ]
    # visualiser.plot_args_resolution(func_example, args_list, add_score=False)
    print(plt.style.available)
    visualiser.style = 'seaborn-deep'
    visualiser.plot_args_resolution(func_example, args_list, add_score=False,
                                    label_list=[
                                        f"{i} * {i} csv file" for i in args_list
                                    ])
