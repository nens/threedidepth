# -*- coding: utf-8 -*-

from unittest import mock
import sys

from threedidepth import commands


def test_threedidepth(tmpdir):
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


def test_threedidepth_with_multiple_steps(tmpdir):
    with mock.patch("threedidepth.commands.calculate_waterdepth") as wd:
        args = ["threedidepth", "a", "b", "c", "d", "--steps", "1", "2", "3"]
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


def test_threediwq(tmpdir):
    with mock.patch("threedidepth.commands.calculate_water_quality") as wq:
        args = ["threediwq", "a", "b", "c", "1", "2", "3", "4", "d"]
        with mock.patch.object(sys, "argv", args):
            commands.threediwq()
        wq.assert_called_with(
            gridadmin_path="a",
            water_quality_results_3di_path="b",
            variable="c",
            output_extent=[1, 2, 3, 4],
            output_path="d",
            calculation_steps=None,
            mode=commands.MODE_LIZARD_VAR,
            calculate_maximum_concentration=False,
            progress_func=None,
            netcdf=False
        )
        args.append("--constant")
        with mock.patch.object(sys, "argv", args):
            commands.threediwq()
        wq.assert_called_with(
            gridadmin_path="a",
            water_quality_results_3di_path="b",
            variable="c",
            output_extent=[1, 2, 3, 4],
            output_path="d",
            calculation_steps=None,
            mode=commands.MODE_CONSTANT_VAR,
            calculate_maximum_concentration=False,
            progress_func=None,
            netcdf=False
        )
        args.append("--maximum")
        with mock.patch.object(sys, "argv", args):
            commands.threediwq()
        wq.assert_called_with(
            gridadmin_path="a",
            water_quality_results_3di_path="b",
            variable="c",
            output_extent=[1, 2, 3, 4],
            output_path="d",
            calculation_steps=None,
            mode=commands.MODE_CONSTANT_VAR,
            calculate_maximum_concentration=True,
            progress_func=None,
            netcdf=False
        )


def test_threediwq_with_multiple_steps(tmpdir):
    with mock.patch("threedidepth.commands.calculate_water_quality") as wq:
        args = ["threediwq", "a", "b", "c", "1", "2", "3", "4", "d"]
        args += ["--steps", "1", "2", "3"]
        with mock.patch.object(sys, "argv", args):
            commands.threediwq()
        wq.assert_called_with(
            gridadmin_path="a",
            water_quality_results_3di_path="b",
            variable="c",
            output_extent=[1, 2, 3, 4],
            output_path="d",
            calculation_steps=[1, 2, 3],
            mode=commands.MODE_LIZARD_VAR,
            calculate_maximum_concentration=False,
            progress_func=None,
            netcdf=False
        )
