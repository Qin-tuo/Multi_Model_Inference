"""Contract checks for native multimodal counting; HTTP responses are fixtures."""
import json
import unittest
import httpx
from app.adapters.llama import LlamaConfig, create_llama_service
from pipecat.processors.aggregators.llm_context import LLMContext
from app.session import PromptBudget
from app.tools import ToolTurn


class NativeBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_prefix_budget_matches_provider_body_without_mutating_context(self):
        turn = ToolTurn("FIXTURE", "十秒后提醒我喝水", None, None)
        context = LLMContext(messages=[{"role": "user", "content": turn.user_text}],
            tools=turn.schemas(), tool_choice="required")
        config = LlamaConfig("fixture", tool_call_prefix="<tool_call>\n")
        service = create_llama_service(config)
        budget = PromptBudget(config.base_url, context_tokens=2048, output_tokens=128, tool_call_prefix=config.tool_call_prefix)
        sent = []
        await budget.client.aclose()
        budget.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request:
            (sent.append(json.loads(request.content)) or httpx.Response(200, json={"input_tokens": 100}))))
        try:
            params = service.build_chat_completion_params(budget.adapter.get_llm_invocation_params(context, convert_developer_to_user=True))
            await budget.count(context)
            self.assertEqual(sent[0]["messages"], params["messages"])
            self.assertIs(sent[0]["parallel_tool_calls"], False)
            self.assertIs(params["parallel_tool_calls"], False)
            self.assertEqual(params["messages"][-1], {"role": "assistant", "content": "<tool_call>\n"})
            self.assertEqual(len(context.get_messages()), 1)
            context.set_tool_choice("auto")
            plain = service.build_chat_completion_params(budget.adapter.get_llm_invocation_params(context, convert_developer_to_user=True))
            self.assertEqual(plain["messages"][-1]["role"], "user")
            self.assertIs(plain["parallel_tool_calls"], False)
        finally:
            await budget.client.aclose()
            await service._client.close()

    async def test_native_template_tool_turn_has_matching_serial_policy(self):
        turn = ToolTurn("FIXTURE", "保存笔记，标题是检查，内容是整理桌面。", None, None, records=object())
        context = LLMContext(messages=[{"role": "user", "content": turn.user_text}],
            tools=turn.schemas(), tool_choice="required")
        config = LlamaConfig("fixture")
        service = create_llama_service(config)
        budget = PromptBudget(config.base_url, context_tokens=2048, output_tokens=128)
        sent = []
        await budget.client.aclose()
        budget.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request:
            (sent.append(json.loads(request.content)) or httpx.Response(200, json={"input_tokens": 100}))))
        try:
            params = service.build_chat_completion_params(budget.adapter.get_llm_invocation_params(context, convert_developer_to_user=True))
            await budget.count(context)
            self.assertIs(params["parallel_tool_calls"], False)
            self.assertIs(sent[0]["parallel_tool_calls"], False)
            self.assertEqual(params["messages"], context.get_messages())
            context.set_tools([])
            params = service.build_chat_completion_params(budget.adapter.get_llm_invocation_params(context, convert_developer_to_user=True))
            self.assertNotIn("parallel_tool_calls", params)
        finally:
            await budget.client.aclose()
            await service._client.close()

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
