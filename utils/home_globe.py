"""Build the BluPrint home globe HTML (no Streamlit). Authentication uses an HttpOnly cookie."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_HOME_GLOBE_TEMPLATE = (_ROOT / "data" / "home_globe.html").resolve()
_AIRPORT_PARQUET = (_ROOT / "data" / "raw" / "airport" / "cirium_airport_ref.parquet").resolve()

HOME_AUTH_COOKIE_NAME = "airside_home_sess"

LOGIN_CREDENTIALS: dict[str, str] = {
    "admin": "admin123",
}


def _auth_secret() -> bytes:
    return (os.environ.get("HOME_AUTH_SECRET") or "airside-demo-secret-change-me").encode("utf-8")


def validate_home_login(username: str, password: str) -> bool:
    if not username or password is None:
        return False
    expected = LOGIN_CREDENTIALS.get(username)
    if expected is None:
        return False
    return hmac.compare_digest(str(password), str(expected))


def build_home_cookie_value(username: str) -> str:
    u = username.encode("utf-8")
    mac = hmac.new(_auth_secret(), u, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(u + b"\n" + mac).decode("ascii")


def _cookie_value(cookie_header: str, name: str) -> str | None:
    for part in cookie_header.split(";"):
        part = part.strip()
        prefix = name + "="
        if part.startswith(prefix):
            return part[len(prefix) :].strip()
    return None


def home_cookie_header_is_valid(cookie_header: str | None) -> bool:
    raw = _cookie_value(cookie_header or "", HOME_AUTH_COOKIE_NAME)
    if not raw:
        return False
    try:
        blob = base64.urlsafe_b64decode(raw.encode("ascii"))
    except (OSError, ValueError, UnicodeEncodeError):
        return False
    if b"\n" not in blob:
        return False
    user_b, mac = blob.split(b"\n", 1)
    try:
        user = user_b.decode("utf-8")
    except UnicodeDecodeError:
        return False
    if user not in LOGIN_CREDENTIALS:
        return False
    exp = hmac.new(_auth_secret(), user_b, hashlib.sha256).digest()
    return hmac.compare_digest(mac, exp)


def _airports_json_for_globe() -> str:
    if not _AIRPORT_PARQUET.is_file():
        return "[]"
    df_airport = pd.read_parquet(_AIRPORT_PARQUET)
    df_airport["airport_name"] = df_airport["name"] + " (" + df_airport["airport_id"] + ")"
    airports_data = [
        {"lat": float(row["lat"]), "lon": float(row["lon"]), "name": str(row["airport_name"])}
        for _, row in df_airport.iterrows()
    ]
    return json.dumps(airports_data, ensure_ascii=False)


def build_home_globe_html(*, authenticated: bool) -> str:
    if not _HOME_GLOBE_TEMPLATE.is_file():
        raise FileNotFoundError(f"Missing template: {_HOME_GLOBE_TEMPLATE}")
    airports_json = _airports_json_for_globe()
    html = _HOME_GLOBE_TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("__AIRPORTS_JSON__", airports_json)
    overlay = _home_auth_overlay_js(authenticated=authenticated)
    if "</body>" in html:
        return html.replace("</body>", overlay + "\n</body>", 1)
    return html + overlay


def _home_auth_overlay_js(*, authenticated: bool) -> str:
    auth_js = "true" if authenticated else "false"
    return f"""
<div id="homeAuthPanel" style="position:fixed;bottom:24px;right:24px;z-index:2147483647;
  min-width:200px;padding:16px;background:rgba(0,0,0,0.45);border-radius:8px;
  font-family:sans-serif;color:rgba(255,255,255,0.9);font-size:0.8rem;">
  <div id="homeLoginBox" style="display:none">
    <p style="font-weight:300;letter-spacing:0.08em;font-size:0.75rem;margin-bottom:10px;">Sign in</p>
    <label style="color:rgba(255,255,255,0.5);font-size:0.75rem;">Username</label><br/>
    <input id="homeUser" type="text" autocomplete="username" placeholder="ID"
      style="width:100%;margin:4px 0 8px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);
      border-radius:6px;color:rgba(255,255,255,0.95);padding:6px 10px;font-size:0.8rem;box-sizing:border-box"/><br/>
    <label style="color:rgba(255,255,255,0.5);font-size:0.75rem;">Password</label><br/>
    <input id="homePass" type="password" autocomplete="current-password" placeholder="••••••••"
      style="width:100%;margin:4px 0 8px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);
      border-radius:6px;color:rgba(255,255,255,0.95);padding:6px 10px;font-size:0.8rem;box-sizing:border-box"/><br/>
    <span id="homeErr" style="color:#f88;font-size:0.75rem;display:none"></span>
    <button type="button" id="homeLoginBtn"
      style="margin-top:8px;width:100%;background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2);
      color:rgba(255,255,255,0.9);border-radius:6px;padding:8px;cursor:pointer;font-size:0.8rem">Login</button>
  </div>
  <div id="homeSignedBox" style="display:none">
    <p style="margin-bottom:8px">Signed in</p>
    <button type="button" id="homeLogoutBtn"
      style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.15);
      color:rgba(255,255,255,0.9);border-radius:6px;padding:8px 12px;cursor:pointer;font-size:0.8rem">Sign out</button>
  </div>
</div>
<script>
(function() {{
  var authed = {auth_js};
  var loginBox = document.getElementById("homeLoginBox");
  var signedBox = document.getElementById("homeSignedBox");
  if (authed) {{ signedBox.style.display = "block"; }} else {{ loginBox.style.display = "block"; }}
  function postJson(url, obj, cb) {{
    var xhr = new XMLHttpRequest();
    xhr.open("POST", url, true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function() {{
      if (xhr.readyState !== 4) return;
      cb(xhr.status, xhr.responseText || "");
    }};
    xhr.send(JSON.stringify(obj));
  }}
  var btn = document.getElementById("homeLoginBtn");
  if (btn) btn.addEventListener("click", function() {{
    var u = (document.getElementById("homeUser") || {{}}).value || "";
    var p = (document.getElementById("homePass") || {{}}).value || "";
    var err = document.getElementById("homeErr");
    postJson("/api/home-auth", {{username: u, password: p}}, function(status) {{
      if (status === 200) location.reload();
      else {{
        if (err) {{ err.style.display = "block"; err.textContent = "Invalid username or password."; }}
      }}
    }});
  }});
  var ob = document.getElementById("homeLogoutBtn");
  if (ob) ob.addEventListener("click", function() {{
    postJson("/api/home-logout", {{}}, function() {{ location.reload(); }});
  }});
}})();
</script>
"""
