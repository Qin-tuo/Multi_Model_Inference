"""Contract checks for native multimodal counting; HTTP responses are fixtures."""
import json
import unittest
import httpx
from pipecat.processors.aggregators.llm_context import LLMContext
from app.session import PromptBudget


class NativeBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_image_body_reaches_native_counter_without_text_only_fallback(self):
        seen = []
        def count(request):
            seen.append((request.url.path, json.loads(request.content)))
            return httpx.Response(200, json={"input_tokens": 422})
        budget = PromptBudget("http://127.0.0.1:8088/v1", context_tokens=2048, output_tokens=128)
        await budget.client.aclose()
        budget.client = httpx.AsyncClient(transport=httpx.MockTransport(count))
        context = LLMContext(messages=[LLMContext.create_image_url_message(url="data:image/png;base64,FIXTURE", text="颜色")])
        try:
            self.assertEqual(await budget.count(context), 422)
            self.assertEqual(seen[0][0], "/v1/chat/completions/input_tokens")
            self.assertIn("data:image/png;base64,FIXTURE", str(seen[0][1]))
        finally: await budget.client.aclose()

    async def test_invalid_provider_count_is_not_a_zero_token_success(self):
        budget = PromptBudget("http://127.0.0.1:8088/v1", context_tokens=2048, output_tokens=128)
        await budget.client.aclose()
        budget.client = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"input_tokens": True})))
        try:
            with self.assertRaises(ValueError): await budget.count(LLMContext())
        finally: await budget.client.aclose()


if __name__ == "__main__": unittest.main()
