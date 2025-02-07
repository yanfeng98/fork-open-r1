import unittest
from open_r1.rewards import accuracy_reward, format_reward, reasoning_steps_reward


class TestRewards(unittest.TestCase):
    def test_accuracy_reward_correct_answer(self):
        """Test accuracy_reward with a correct answer."""
        completion = [[{"content": r"\boxed{\frac{63}{400}}"}]]
        solution = [r"\frac{63}{400}"]
        
        rewards = accuracy_reward(completion, solution)
        self.assertEqual(rewards[0], 1.0)

    def test_accuracy_reward_wrong_answer(self):
        """Test accuracy_reward with an incorrect answer."""
        completion = [[{"content": r"\boxed{\frac{64}{400}}"}]]
        solution = [r"\frac{63}{400}"]
        
        rewards = accuracy_reward(completion, solution)
        self.assertEqual(rewards[0], 0.0)

    def test_format_reward_correct(self):
        """Test format_reward with correct format."""
        completion = [[{"content": "<think>Some reasoning</think><answer>The answer</answer>"}]]
        rewards = format_reward(completion)
        self.assertEqual(rewards[0], 1.0)

    def test_format_reward_incorrect(self):
        """Test format_reward with incorrect format."""
        incorrect_formats = [
            "<think>Only thinking</think>",
            "<answer>Only answer</answer>",
            "No tags at all",
            "<think>Missing closing</think><answer>Missing closing",
            "<think>Wrong order</answer><answer>Wrong order</think>"
        ]
        
        for fmt in incorrect_formats:
            completion = [[{"content": fmt}]]
            rewards = format_reward(completion)
            self.assertEqual(rewards[0], 0.0)

    def test_reasoning_steps_reward(self):
        """Test reasoning_steps_reward with various formats."""
        test_cases = [
            # Full credit cases (3 or more steps)
            (
                "Step 1: First step\nStep 2: Second step\nStep 3: Third step",
                1.0
            ),
            (
                "First, we do this.\nSecond, we do that.\nFinally, we conclude.",
                1.0
            ),
            # Partial credit cases (less than 3 steps)
            (
                "Step 1: Only step",
                1/3
            ),
            (
                "First, we do this.\nFinally, we conclude.",
                2/3
            ),
            # No credit case
            (
                "Just plain text without any clear steps",
                0.0
            )
        ]
        
        for content, expected_reward in test_cases:
            completion = [[{"content": content}]]
            rewards = reasoning_steps_reward(completion)
            self.assertAlmostEqual(rewards[0], expected_reward)

    def test_multiple_completions(self):
        """Test handling multiple completions at once."""
        completions = [
            [{"content": r"\boxed{\frac{63}{400}}"}],
            [{"content": r"\boxed{\frac{64}{400}}"}]
        ]
        solutions = [r"\frac{63}{400}", r"\frac{63}{400}"]
        
        rewards = accuracy_reward(completions, solutions)
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 0.0)


if __name__ == '__main__':
    unittest.main() 