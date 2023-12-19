# -*- coding: utf-8 -*-
"""
Tests to check if runs correctly.

These tests will run in the Jenkins pipeline after each change
made to the code.
"""

import unittest

import tbbrdet_api.api as api


class TestModelMethods(unittest.TestCase):
    def setUp(self):
        self.meta = api.get_metadata()

    def test_model_metadata_type(self):
        """
        Test that get_metadata() returns dict
        """
        self.assertTrue(type(self.meta) is dict)

    def test_model_metadata_values(self):
        """
        Test that get_metadata() returns right values (subset)
        """
        self.assertEqual(
            self.meta["model_name"].lower(), "TBBRDet".lower()
        )
        self.assertEqual(
            self.meta["api_name"].lower().replace("-", "_"),
            "tbbrdet_api".lower().replace("-", "_"),
        )
        self.assertEqual(
            self.meta["api_authors"].lower(), "Elena Vollmer".lower()
        )
        self.assertEqual(
            self.meta["model_authors"].lower(), "James Kahn".lower()
        )
        self.assertEqual(
            self.meta["license"].lower(), "BSD-3-Clause".lower(),
        )
        self.assertIn(
            "0.0.1", self.meta["version"]
        )


if __name__ == "__main__":
    unittest.main()
