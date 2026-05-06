from __future__ import annotations

import os
from typing import Optional

from utils.layout_design_build import build_layout_design_html


def build_designer_html(
    *,
    layout_api_url: str,
    grid3d_asset_api_url: str,
    load_layout: Optional[str] = None,
) -> str:
    """Build the Layout Design document without importing Streamlit."""
    old_env = {
        "LAYOUT_SAME_PORT": os.environ.get("LAYOUT_SAME_PORT"),
        "LAYOUT_API_BASE_URL": os.environ.get("LAYOUT_API_BASE_URL"),
        "GRID3D_ASSET_API_URL": os.environ.get("GRID3D_ASSET_API_URL"),
    }
    try:
        os.environ["LAYOUT_SAME_PORT"] = "1"
        os.environ["LAYOUT_API_BASE_URL"] = layout_api_url
        os.environ["GRID3D_ASSET_API_URL"] = grid3d_asset_api_url
        html = build_layout_design_html(
            layout_api_url=layout_api_url,
            grid3d_asset_api_url=grid3d_asset_api_url,
            load_layout=load_layout,
        )
        if not isinstance(html, str) or not html.strip():
            raise RuntimeError("Layout Design HTML build produced empty output")
        return html
    finally:
        for key, val in old_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
