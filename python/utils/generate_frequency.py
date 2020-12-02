import pytuning
from pytuning.tuning_tables import create_timidity_tuning

import sympy as sp

import argparse
import os

def write_tuning_table(root_dir):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    os.chdir(root_dir)
    scale = pytuning.create_edo_scale(12)
    tuning = create_timidity_tuning(scale, reference_note=60)
    with open("equal.txt", "w") as table:
        table.write(tuning)
    print("Generating 12-equal using: ")
    print(scale)
    scale = pytuning.create_pythagorean_scale(number_down_fifths=0)
    tuning = create_timidity_tuning(scale, reference_note=60)
    with open("pythagorean.txt", "w") as table:
        table.write(tuning)
    print("Generating pythagorean using: ")
    print(scale)

    scale = [
        sp.Integer(1), # C
        sp.Rational(16, 15), # C#
        sp.Rational(9, 8), # D
        sp.Rational(6, 5), # D#
        sp.Rational(5, 4), # E
        sp.Rational(4, 3), # F
        sp.Rational(45, 32), # F#
        sp.Rational(3, 2), # G
        sp.Rational(8, 5), # G#
        sp.Rational(5, 3), # A
        sp.Rational(9, 5), # A#
        sp.Rational(15, 8), # B
        sp.Integer(2), # C
    ] # see https://wenku.baidu.com/view/d71e97520a1c59eef8c75fbfc77da26925c59634.html
    tuning = create_timidity_tuning(scale, reference_note=60)
    with open("pure.txt", "w") as table:
        table.write(tuning)
    print("Generating pure using: ")
    print(scale)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", required=True, help="Path to place the requency table")
    args = parser.parse_args()
    write_tuning_table(args.path)
