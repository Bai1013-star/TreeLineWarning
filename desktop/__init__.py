"""PyWebView desktop shell for the existing local Giro3D viewer."""

from desktop.corridor_extensions import install_corridor_extensions
from desktop.review_extensions import install_review_extensions

install_review_extensions()
install_corridor_extensions()
