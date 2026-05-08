"""Smoke tests for CDI-ST — ensure the package imports and basic objects exist."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_version_string_is_valid():
    """Package must declare a version."""
    import cdi_st
    assert cdi_st.__version__
    # Crude PEP 440 check
    parts = cdi_st.__version__.split(".")
    assert len(parts) >= 2


def test_main_entry_point_exists():
    """The console_scripts entry point points to bcdi_gui:main."""
    from cdi_st import bcdi_gui
    assert hasattr(bcdi_gui, "main")
    assert callable(bcdi_gui.main)


def test_launcher_class_exists():
    """The Launcher class is defined and has the expected windows."""
    from cdi_st.bcdi_gui import Launcher, MW_Sim, MW_Analysis
    assert Launcher is not None
    assert MW_Sim is not None
    assert MW_Analysis is not None


def test_logo_packaged():
    """The CDI-ST logo PNG must be discoverable inside the installed package."""
    from cdi_st.bcdi_gui import _logo_path
    p = _logo_path()
    assert p is not None, "Logo not found in installed package directory"
    assert os.path.exists(p)


def test_reports_dialog_class():
    """ReportsDialog is exposed and points to the maintainer email."""
    from cdi_st.bcdi_gui import ReportsDialog
    assert ReportsDialog.RECIPIENT == "saidisoufiane@hotmail.com"
