"""코드 가이드 페이지 — docs/CODE_GUIDE.md를 읽어 화면에 표시한다.

문서 원본은 `docs/CODE_GUIDE.md` 하나뿐이다(단일 원본). 이 페이지는 그 파일을
읽어 렌더링만 하므로 문서와 화면이 어긋나지 않는다.
"""
from __future__ import annotations

import os

import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUIDE_PATH = os.path.join(ROOT, "docs", "CODE_GUIDE.md")


@st.cache_data(show_spinner=False)
def _load_guide(path: str, mtime: float) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _split_sections(md: str) -> list[tuple[str, str]]:
    """'## ' 기준으로 (제목, 본문) 목록으로 나눈다. 앞부분(머리말)은 '개요'로."""
    lines = md.splitlines()
    sections: list[tuple[str, list[str]]] = [("개요", [])]
    for ln in lines:
        if ln.startswith("## "):
            sections.append((ln[3:].strip(), []))
        else:
            sections[-1][1].append(ln)
    return [(t, "\n".join(b).strip()) for t, b in sections if "\n".join(b).strip()]


def render_code_guide():
    st.header("🧩 코드 가이드 (개발자 문서)")
    if not os.path.exists(GUIDE_PATH):
        st.error(f"문서를 찾을 수 없습니다: {GUIDE_PATH}")
        return

    md = _load_guide(GUIDE_PATH, os.path.getmtime(GUIDE_PATH))
    sections = _split_sections(md)
    titles = [t for t, _ in sections]

    st.caption("소스코드 구조와 동작 원리를 설명합니다. "
               "원본은 `docs/CODE_GUIDE.md` — 에디터·GitHub에서도 같은 내용을 볼 수 있습니다.")

    c1, c2 = st.columns([3, 1])
    with c1:
        pick = st.selectbox("섹션 이동", titles, key="code_section")
    with c2:
        st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
        show_all = st.checkbox("전체 보기", key="code_all")
    with open(GUIDE_PATH, "rb") as f:
        st.download_button("📥 CODE_GUIDE.md 내려받기", f.read(),
                           "CODE_GUIDE.md", "text/markdown")
    st.divider()

    if show_all:
        st.markdown(md)
    else:
        body = dict(sections)[pick]
        if pick != "개요":
            st.markdown(f"## {pick}")
        st.markdown(body)
