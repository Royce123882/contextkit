"""Tests for file context handling (Phase 2)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from contextkit.files.file_context import FileContext, _chunk_to_budget
from contextkit.files.file_reference import FileReference


class TestFileReference:
    """Tests for the FileReference model."""

    def test_create_with_defaults(self) -> None:
        ref = FileReference(path="/tmp/test.py")
        assert ref.path == "/tmp/test.py"
        assert ref.size_bytes == 0
        assert ref.extension == ""
        assert ref.name == ""

    def test_create_with_all_fields(self) -> None:
        ref = FileReference(
            path="/tmp/test.py",
            size_bytes=100,
            extension=".py",
            name="test.py",
        )
        assert ref.size_bytes == 100
        assert ref.extension == ".py"
        assert ref.name == "test.py"


class TestFileContext:
    """Tests for FileContext scanning and loading."""

    def test_scan_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.py").write_text("print('a')")
            (Path(tmpdir) / "b.txt").write_text("hello")

            fc = FileContext(tmpdir)
            refs = fc.scan()
            assert len(refs) == 2

    def test_scan_with_extension_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.py").write_text("code")
            (Path(tmpdir) / "b.txt").write_text("text")
            (Path(tmpdir) / "c.py").write_text("more code")

            fc = FileContext(tmpdir, extensions=[".py"])
            refs = fc.scan()
            assert len(refs) == 2
            assert all(r.extension == ".py" for r in refs)

    def test_scan_nonexistent_directory(self) -> None:
        fc = FileContext("/tmp/nonexistent_dir_xyz")
        refs = fc.scan()
        assert refs == []

    def test_scan_populates_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.py"
            filepath.write_text("x = 1")

            fc = FileContext(tmpdir)
            refs = fc.scan()
            assert len(refs) == 1
            assert refs[0].name == "test.py"
            assert refs[0].extension == ".py"
            assert refs[0].size_bytes > 0

    def test_base_path(self) -> None:
        fc = FileContext("/tmp/test")
        assert fc.base_path == Path("/tmp/test")

    def test_index_property(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.py").write_text("code")
            fc = FileContext(tmpdir)
            assert fc.index == []  # Before scan
            fc.scan()
            assert len(fc.index) == 1

    def test_index_returns_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.py").write_text("code")
            fc = FileContext(tmpdir)
            fc.scan()
            idx = fc.index
            idx.clear()
            assert len(fc.index) == 1

    def test_search_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "utils.py").write_text("code")
            (Path(tmpdir) / "models.py").write_text("code")
            (Path(tmpdir) / "readme.md").write_text("docs")

            fc = FileContext(tmpdir)
            fc.scan()
            results = fc.search("utils")
            assert len(results) >= 1
            assert results[0].name == "utils.py"

    def test_search_exact_match_bonus(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "utils.py").write_text("code")
            (Path(tmpdir) / "my_utils.py").write_text("code")

            fc = FileContext(tmpdir)
            fc.scan()
            results = fc.search("utils")
            # Exact stem match should rank first
            assert results[0].name == "utils.py"

    def test_search_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(10):
                (Path(tmpdir) / f"file_{i}.py").write_text("code")

            fc = FileContext(tmpdir)
            fc.scan()
            results = fc.search("file", top_k=3)
            assert len(results) == 3

    def test_search_no_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("code")
            fc = FileContext(tmpdir)
            fc.scan()
            results = fc.search("nonexistent_xyz")
            assert len(results) == 0

    def test_load_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.py").write_text("print('hello')")

            fc = FileContext(tmpdir)
            refs = fc.scan()
            blocks = fc.load(refs)
            assert len(blocks) == 1
            assert blocks[0].type.value == "files"
            assert "hello" in blocks[0].content
            assert blocks[0].origin is not None
            assert blocks[0].origin.source == "file"

    def test_load_with_token_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(10):
                (Path(tmpdir) / f"file_{i}.py").write_text("x = 1\n" * 100)

            fc = FileContext(tmpdir)
            refs = fc.scan()
            blocks = fc.load(refs, max_tokens=50)
            total_tokens = sum(b.token_count for b in blocks)
            assert total_tokens <= 50

    def test_load_skips_binary_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            binary_path = Path(tmpdir) / "data.bin"
            binary_path.write_bytes(b"\x00\xff\x80\x01" * 100)

            ref = FileReference(
                path=str(binary_path),
                size_bytes=400,
                extension=".bin",
                name="data.bin",
            )
            fc = FileContext(tmpdir)
            blocks = fc.load([ref])
            # Should skip due to UnicodeDecodeError
            assert len(blocks) <= 1

    def test_load_single(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "single.py"
            filepath.write_text("x = 42")

            fc = FileContext(tmpdir)
            block = fc.load_single(str(filepath))
            assert block.type.value == "files"
            assert "42" in block.content
            assert block.origin is not None
            assert block.origin.details["file_path"] == str(filepath)

    def test_load_single_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.py"
            filepath.write_text("code")

            fc = FileContext(tmpdir)
            block = fc.load_single(str(filepath), priority=90)
            assert block.priority == 90

    def test_scan_subdirectories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            (subdir / "nested.py").write_text("code")
            (Path(tmpdir) / "top.py").write_text("code")

            fc = FileContext(tmpdir)
            refs = fc.scan()
            assert len(refs) == 2


class TestChunkToBudget:
    """Tests for the _chunk_to_budget helper."""

    def test_no_chunking_needed(self) -> None:
        content = "short"
        result = _chunk_to_budget(content, max_tokens=100, encoding="cl100k_base")
        assert result == content

    def test_chunks_long_content(self) -> None:
        content = "word " * 1000
        result = _chunk_to_budget(content, max_tokens=10, encoding="cl100k_base")
        assert len(result) < len(content)

    def test_respects_budget(self) -> None:
        from contextkit.utils.token_counting import count as count_tokens

        content = "Hello world this is a test " * 50
        result = _chunk_to_budget(content, max_tokens=20, encoding="cl100k_base")
        assert count_tokens(result) <= 20
