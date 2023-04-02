#!/usr/bin/env python3
"""
Version 2 uses subprocess to run Part1, Part2, and Part3 in sequence for testing purposes.
"""

import subprocess

subprocess.run(["python3", "mainPart1_webscrape.py"])
subprocess.run(["python3", "mainPart2_wordEmbedAndProcess.py"])
subprocess.run(["python3", "mainPart3_inputClusterAndRankingGenerator.py"])
