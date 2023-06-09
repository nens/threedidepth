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
            calculation_steps=None,
            mode=commands.MODE_LIZARD,
            calculate_maximum_waterlevel=False,
            progress_func=None,
            netcdf=False
        )
        args.append("--constant")
        with mock.patch.object(sys, "argv", args):
            commands.threedidepth()
        wd.assert_called_with(
            gridadmin_path="a",
            results_3di_path="b",
            dem_path="c",
            waterdepth_path="d",
            calculation_steps=None,
            mode=commands.MODE_CONSTANT,
            calculate_maximum_waterlevel=False,
            progress_func=None,
            netcdf=False
        )
        args.append("--maximum")
        with mock.patch.object(sys, "argv", args):
            commands.threedidepth()
        wd.assert_called_with(
            gridadmin_path="a",
            results_3di_path="b",
            dem_path="c",
            waterdepth_path="d",
            calculation_steps=None,
            mode=commands.MODE_CONSTANT,
            calculate_maximum_waterlevel=True,
            progress_func=None,
            netcdf=False
        )


def test_command_with_multiple_steps(tmpdir):
    depth_path = tmpdir.join("waterdepth.tif")
    depth_path.ensure(file=True)  # "touch" the file
    with mock.patch("threedidepth.commands.calculate_waterdepth") as wd:
        args = ["threedidepth,", "a", "b", "c", "d", "--steps", "1", "2", "3"]
        with mock.patch.object(sys, "argv", args):
            commands.threedidepth()
        wd.assert_called_with(
            gridadmin_path="a",
            results_3di_path="b",
            dem_path="c",
            waterdepth_path="d",
            calculation_steps=[1, 2, 3],
            mode=commands.MODE_LIZARD,
            calculate_maximum_waterlevel=False,
            progress_func=None,
            netcdf=False
        )
