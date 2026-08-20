import tempfile
import unittest
from unittest.mock import patch

from utils import knowledge_graph as kg


class KnowledgeGraphTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.graph_dir = patch.object(kg, "GRAPH_DIR", self.temp_dir.name)
        self.graph_dir.start()

    def tearDown(self):
        self.graph_dir.stop()
        self.temp_dir.cleanup()

    def test_two_hop_evidence_keeps_provenance(self):
        kg.build_or_update_graph([
            {"source": "a.pdf", "chunk_id": 0, "text": "A公司与B公司存在供应商关系。"},
            {"source": "b.pdf", "chunk_id": 1, "text": "B公司成本上升会影响毛利率。"},
        ], "session")

        chunks, paths = kg.query_graph(
            "A公司成本变化如何影响毛利率？",
            "session",
            [{"source": "a.pdf", "chunk_id": 0}],
            max_hops=2,
            top_k=5,
        )

        self.assertEqual({c["source"] for c in chunks}, {"a.pdf", "b.pdf"})
        self.assertTrue(all("graph_path" in c for c in chunks))
        self.assertTrue(paths)

    def test_plain_lookup_does_not_trigger_graph(self):
        self.assertFalse(kg.should_use_graph("A公司的营收是多少？"))
        self.assertTrue(kg.should_use_graph("A公司与B公司的关系是什么？"))

    def test_delete_source_removes_graph_records(self):
        kg.build_or_update_graph([
            {"source": "a.pdf", "chunk_id": 0, "text": "A公司收入增长。"},
            {"source": "b.pdf", "chunk_id": 0, "text": "B公司收入下降。"},
        ], "session")
        remaining = kg.delete_source_from_graph("a.pdf", "session")
        self.assertEqual(remaining, 1)
        self.assertEqual(kg.load_graph("session")["records"][0]["source"], "b.pdf")


if __name__ == "__main__":
    unittest.main()
