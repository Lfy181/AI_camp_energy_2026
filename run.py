#!/usr/bin/env python
"""
快捷入口：运行消融实验
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.run_experiments import run_all_experiments

if __name__ == '__main__':
    run_all_experiments()
