from claude_mem.bulk import parse_markdown_doc

DOC = """# Title

Intro paragraph about nothing.

## The April-9 GW design

It was proposed and never built. This is the load-bearing fact.

## A later section

Unrelated content here.
"""


def test_doc_splits_into_sections(tmp_path):
    p = tmp_path / "d.md"
    p.write_text(DOC, encoding="utf-8")
    chunks = parse_markdown_doc(p)
    assert isinstance(chunks, list)
    assert len(chunks) >= 3
    gw = [c for c in chunks if "GW design" in c["content"]]
    assert len(gw) == 1
    assert "never built" in gw[0]["content"]
    assert gw[0]["line_start"] >= 1 and gw[0]["line_end"] >= gw[0]["line_start"]
    assert gw[0]["file_path"].endswith("d.md")


def test_non_utf8_file_does_not_crash(tmp_path):
    from claude_mem.bulk import parse_markdown_doc
    p = tmp_path / "bad.md"
    p.write_bytes(b"# Title\n\n## Sec\n\nan em\x97dash and arrow\n")  # 0x97 = cp1252
    chunks = parse_markdown_doc(p)  # must NOT raise
    assert len(chunks) >= 1
    assert any("�" in c["content"] for c in chunks)  # replacement char present


def test_windowed_sections_have_distinct_line_bounds(tmp_path):
    from claude_mem.bulk import parse_markdown_doc
    big = "\n".join(f"line {i} aaaaaaaaaaaaaaaaaaaa" for i in range(400))  # > 6000 chars
    p = tmp_path / "big.md"
    p.write_text(f"# T\n\n## Big\n\n{big}\n", encoding="utf-8")
    chunks = parse_markdown_doc(p)
    bigs = [c for c in chunks if c["line_start"] >= 3]  # the ## Big section windows
    assert len(bigs) >= 2  # windowed into 2+ chunks
    starts = [c["line_start"] for c in bigs]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)  # strictly increasing, distinct
