import functools
import time
from dataclasses import dataclass

import typing

from memory_profiler import memory_usage
import numpy
import matplotlib.pyplot as plt


def memory_time_dependence(func, args=None, kws=None, interval: float = 0.01,
                           include_children: bool = False):
    if args is None:
        args = ()
    if kws is None:
        kws = {}
    mem_usage = memory_usage((func, args, kws), interval=interval,
                             include_children=include_children)
    return numpy.linspace(0, len(mem_usage) * interval,
                          len(mem_usage)), mem_usage


def time_memory_of_execution(func, args=None, kws=None, interval: float = 0.01,
                             include_children: bool = False):
    t, mem = memory_time_dependence(func, args, kws, interval,
                                    include_children)
    return t[-1], max(mem)


def timeit_memit(func, args=None, kws=None, interval: float = 0.01,
                 include_children: bool = False, nums_repeats=4, ):
    tcc = float(str("inf"))
    mcc = -float(str("inf"))
    for _ in range(nums_repeats):
        t, m = time_memory_of_execution(func, args, kws, interval,
                                        include_children)
        tcc = min(t, tcc)
        mcc = max(m, mcc)
    return tcc, mcc


def args_kws_resolution(func, args_list: typing.List[tuple] = None,
                        kws_list: typing.List[tuple] = None,
                        interval: float = 0.01,
                        include_children: bool = False, mode="usage"):
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
        return zip(*args_kws_resolution(func, args_list, kws_list,
                                        self.interval,
                                        self.include_children,
                                        mode="time_resolve"))

    def plot_args_resolution(self, func, args_list=None, kws_list=None,
                             label_list=None,
                             title=None):
        kws_list = kws_list or [{}] * len(args_list)
        label_list = label_list or [f"args={args}\nkws={kws}" for args, kws in
                                    zip(args_list, kws_list)]
        title = title or f"{func.__name__} args_kws resolution"
        t_usage = []
        m_usage = []

        plt.style.use(self.style)
        fig, (ax_1, ax_2) = plt.subplots(nrows=2, ncols=1)
        for idx, (t, mem) in enumerate(
                self.get_args_kws_dependence(func, args_list, kws_list)
        ):
            ax_1.plot(t, mem, label=label_list[idx], linestyle="--")
            t_usage.append(t[-1])
            m_usage.append(max(mem))

        ax_1.legend()
        ax_2.scatter(t_usage, m_usage)
        for x, y, txt in zip(t_usage, m_usage, label_list):
            ax_2.annotate(txt, (x, y))
        ax_1.set_xlabel(self.time_label)
        ax_2.set_xlabel(self.time_label)
        ax_1.set_ylabel(self.memory_label)
        ax_2.set_ylabel(self.memory_label)
        ax_1.set_title(title)
        plt.tight_layout()
        if self.show:
            plt.show()

    def plot_memory_time(self, time_list, memory_list, label, title,
                         annotations=None):
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

    def plot_memory_time_resolution(self, func, args=None, kws=None,
                                    label=None, title=None):
        time_list, memory_list = memory_time_dependence(func, args, kws,
                                                        interval=self.interval,
                                                        include_children=self.include_children, )
        label = label or f"{func.__name__}\nargs={args}\nkws={kws}"
        title = title or f"{func.__name__} memory-time resolution"
        self.plot_memory_time(time_list, memory_list, label, title)


if __name__ == "__main__":

    def foo(**kwargs):
        time.sleep(0.5)


    def func_example(*args, **kwargs):
        foo(**kwargs)
        if args:
            time.sleep(min(len(args), 1))


    # print(f"time_memory_of_execution", time_memory_of_execution(func_example))
    # print(f"timit memit", timeit_memit(func_example, nums_repeats=10))
    # print("timer", timer(func_example)())
    # args_list = [(1, 2), (3, 4), (5, 6), tuple(range(10 ** 6))]
    # print("args_resol",
    #       args_kws_resolution(func_example, args_list, mode="usage"))
    # print("args_resol_timr",
    #       args_kws_resolution(func_example, args_list, mode="time_resolve"))
    # print()
    # print(memory_time_dependence(func_example, args=(3, 4)))

    # visualiser = ProfileVisualiser()
    # visualiser.plot_memory_time_resolution(func_example, args=(1, 2, 3))
    # visualiser.plot_args_resolution(func_example, args_list=[(1, 2), (1, 2, 3),
    #                                                          (1, 2, 3, 4),
    #                                                          (1,)])
    # print(visualiser._res)
