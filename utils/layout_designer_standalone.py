from __future__ import annotations

import os
import runpy
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional


_ROOT = Path(__file__).resolve().parents[1]
_LAYOUT_PAGE = _ROOT / "pages" / "3_⚙️ Layout_Design.py"


class _FakeQueryParams(dict):
    def get(self, key: str, default: Any = None) -> Any:
        return super().get(key, default)


def _fake_streamlit_modules(load_layout: Optional[str]) -> Dict[str, types.ModuleType]:
    st_mod = types.ModuleType("streamlit")
    st_mod.query_params = _FakeQueryParams(
        {"load_layout": load_layout} if load_layout else {}
    )

    def _noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    def _experimental_get_query_params() -> Dict[str, list[str]]:
        return {"load_layout": [load_layout]} if load_layout else {}

    st_mod.set_page_config = _noop  # type: ignore[attr-defined]
    st_mod.markdown = _noop  # type: ignore[attr-defined]
    st_mod.experimental_get_query_params = _experimental_get_query_params  # type: ignore[attr-defined]

    components_pkg = types.ModuleType("streamlit.components")
    components_v1 = types.ModuleType("streamlit.components.v1")

    def _html(component_html: str, *_args: Any, **_kwargs: Any) -> None:
        st_mod.__standalone_component_html__ = component_html  # type: ignore[attr-defined]

    components_v1.html = _html  # type: ignore[attr-defined]
    components_pkg.v1 = components_v1  # type: ignore[attr-defined]
    st_mod.components = components_pkg  # type: ignore[attr-defined]

    return {
        "streamlit": st_mod,
        "streamlit.components": components_pkg,
        "streamlit.components.v1": components_v1,
    }


def build_designer_html(
    *,
    layout_api_url: str,
    grid3d_asset_api_url: str,
    load_layout: Optional[str] = None,
) -> str:
    """Build the Layout Design document without importing the real Streamlit runtime."""
    if not _LAYOUT_PAGE.is_file():
        raise FileNotFoundError(f"Layout Design page missing: {_LAYOUT_PAGE}")

    fake_modules = _fake_streamlit_modules(load_layout)
    old_modules = {name: sys.modules.get(name) for name in fake_modules}
    old_env = {
        "LAYOUT_SAME_PORT": os.environ.get("LAYOUT_SAME_PORT"),
        "LAYOUT_API_BASE_URL": os.environ.get("LAYOUT_API_BASE_URL"),
        "GRID3D_ASSET_API_URL": os.environ.get("GRID3D_ASSET_API_URL"),
    }
    try:
        sys.modules.update(fake_modules)
        os.environ["LAYOUT_SAME_PORT"] = "1"
        os.environ["LAYOUT_API_BASE_URL"] = layout_api_url
        os.environ["GRID3D_ASSET_API_URL"] = grid3d_asset_api_url
        ns = runpy.run_path(str(_LAYOUT_PAGE), run_name="__layout_design_standalone__")
        html = ns.get("html")
        if not isinstance(html, str) or not html.strip():
            captured = getattr(
                fake_modules["streamlit"], "__standalone_component_html__", None
            )
            html = captured if isinstance(captured, str) else ""
        if not html.strip():
            raise RuntimeError("Layout Design HTML build produced empty output")
        return html
    finally:
        for key, val in old_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
        for name, old in old_modules.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
