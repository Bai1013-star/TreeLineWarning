"""PyInstaller runtime hook for the bundled PDAL native libraries."""

from desktop.pdal_runtime import _configure_bundled_runtime

_configure_bundled_runtime()
