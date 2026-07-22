import os
import sys
import unittest

# Add src and scripts to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import format_instruction_prompt
from scripts.eval import clean_prediction

class TestUtils(unittest.TestCase):
    def test_prompt_formatter(self):
        instruction = "How can I cancel my order?"
        intents = ["cancel_order", "track_package"]
        
        # Test with intent list
        prompt_with_list = format_instruction_prompt(instruction, intents)
        self.assertIn("cancel_order, track_package", prompt_with_list)
        self.assertIn("Query: How can I cancel my order?", prompt_with_list)
        self.assertTrue(prompt_with_list.endswith("intent:"))
        
        # Test without intent list
        prompt_no_list = format_instruction_prompt(instruction)
        self.assertNotIn("cancel_order", prompt_no_list)
        self.assertIn("Query: How can I cancel my order?", prompt_no_list)

    def test_label_parser(self):
        intents = ["cancel_order", "track_package", "refund_request"]
        
        # Exact match
        self.assertEqual(clean_prediction("cancel_order", intents), "cancel_order")
        self.assertEqual(clean_prediction("  track_package  \n", intents), "track_package")
        
        # Substring match
        self.assertEqual(clean_prediction("The intent is refund_request", intents), "refund_request")
        
        # Case insensitive
        self.assertEqual(clean_prediction("CANCEL_ORDER", intents), "cancel_order")
        
        # Fallback
        self.assertEqual(clean_prediction("some random text", intents), "cancel_order")

if __name__ == "__main__":
    unittest.main()
