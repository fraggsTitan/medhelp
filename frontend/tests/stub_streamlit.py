"""A tiny stand-in for the ``streamlit`` module so app.py can be executed (and asserted on) without Streamlit."""
from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from typing import Any


class Rerun(Exception):
    """What ``st.rerun()`` does to the running script."""


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Col(_Ctx):
    def __init__(self, stub):
        self.stub = stub

    def button(self, *a, **kw):
        return self.stub.button(*a, **kw)


class _Empty:
    def __init__(self, stub):
        self.stub = stub

    def caption(self, text, **kw):
        self.stub.log("empty.caption", text)

    def empty(self):
        self.stub.log("empty.cleared")

    markdown = caption


class Audio:
    type = "audio/wav"

    def __init__(self, data=b"RIFFaudio"):
        self.data = data

    def getvalue(self):
        return self.data


class Stub(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state: dict[str, Any] = {}
        self.query_params: dict[str, str] = {}
        self.secrets: dict[str, str] = {}
        self.sidebar = _Ctx()
        self.calls: list[tuple[str, Any]] = []
        self.clicked: set[str] = set()
        self.typed: Any = None
        self.audio: Any = None
        self.streamed: list[str] = []

    def log(self, name, value=None):
        self.calls.append((name, value))

    def texts(self, name):
        return [str(v) for n, v in self.calls if n == name]

    # ---- layout / text ----
    def set_page_config(self, **kw): self.log("set_page_config", kw)
    def markdown(self, text, **kw): self.log("markdown", text)
    def caption(self, text, **kw): self.log("caption", text)
    def write(self, *a, **kw): self.log("write", a)
    def divider(self): pass
    def json(self, obj): self.log("json", obj)
    def success(self, text, **kw): self.log("success", text)
    def info(self, text, **kw): self.log("info", text)
    def warning(self, text, **kw): self.log("warning", text)
    def error(self, text, **kw): self.log("error", text)
    def chat_message(self, *a, **kw): return _Ctx()
    def spinner(self, *a, **kw): return _Ctx()
    def expander(self, label, **kw):
        self.log("expander", label)
        return _Ctx()
    def columns(self, n): return [_Col(self) for _ in range(n)]
    def empty(self): return _Empty(self)

    # ---- widgets ----
    def button(self, label, key=None, on_click=None, **kw):
        self.log("button", label)
        hit = (key in self.clicked) or (label in self.clicked)
        if hit and on_click:
            on_click()
        return hit

    def toggle(self, label, key=None, **kw):
        self.session_state.setdefault(key, False)
        return self.session_state[key]

    def text_area(self, label, value="", key=None, **kw):
        self.log("text_area", key)
        if key is not None:
            self.session_state.setdefault(key, value)
            return self.session_state[key]
        return value

    def multiselect(self, label, options, key=None, **kw):
        self.log("multiselect", options)
        return list(self.session_state.get(key, []))

    def chat_input(self, *a, **kw): return self.typed
    def audio_input(self, *a, **kw): return self.audio
    def download_button(self, label, **kw):
        self.log("download_button", label)
        return False

    def write_stream(self, gen):
        text = "".join(gen)
        self.streamed.append(text)
        return text

    def rerun(self): raise Rerun()

    # ---- caching decorators are pass-through ----
    def cache_resource(self, *a, **kw):
        return a[0] if a and callable(a[0]) else (lambda f: f)

    cache_data = cache_resource


@contextmanager
def installed(stub: Stub):
    old = sys.modules.get("streamlit")
    sys.modules["streamlit"] = stub
    try:
        yield stub
    finally:
        if old is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = old
