# -*- coding: utf-8 -*-

from unittest import mock
import sys

from threedidepth import commands


def test_command(tmpdir):
    depth_path = tmpdir.join("waterdepth.tif")
    depth_path.ensure(file=True)  # "touch" the file
    with mock.patch("threedidepth.commands.calculate_waterdepth") as wd:
        args = ["threedidepth,", "a", "b", "c", "d"]
        with mock.patch.object(sys, "argv", args):
            commands.threedidepth()
        wd.assert_called_with(
            gridadmin_path="a",
            results_3di_path="b",
            dem_path="c",
            waterdepth_path="d",
            calculation_step=-1,
            mode=commands.MODE_LIZARD,
            progress_func=None,
        )
        args.append("--constant")
        with mock.patch.object(sys, "argv", args):
            commands.threedidepth()
        wd.assert_called_with(
            gridadmin_path="a",
            results_3di_path="b",
            dem_path="c",
            waterdepth_path="d",
            calculation_step=-1,
            mode=commands.MODE_CONSTANT,
            progress_func=None,
        )
