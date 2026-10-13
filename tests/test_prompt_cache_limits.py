"""Both real launcher paths must bound native host prompt caches explicitly."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import project


class PromptCacheLimitTests(unittest.TestCase):
    def test_both_commands_keep_context_and_forward_distinct_cache_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'configs').mkdir();(root/'models').mkdir();(root/'build/llama/bin').mkdir(parents=True)
            (root/'build/llama/bin/llama-server').touch()
            for name in ('vlm.gguf','mmproj-vlm.gguf','asr.gguf','mmproj-asr.gguf'):(root/'models'/name).touch()
            config={'llama':{'prompt_cache_ram_mib':256,'gpu_layers':99,'context_tokens':2048,'threads':4,'parallel':1,'port':8088},
                'speech':{'asr_prompt_cache_ram_mib':64}}
            (root/'configs/runtime.yaml').write_text(json.dumps(config))
            models={'models':{name:{'files':[{'name':name+'.gguf'},{'name':'mmproj-'+name+'.gguf'}],
                'context_tokens':2048,'port':8089} for name in ('vlm','asr')}}
            with patch.object(project,'ROOT',root),patch.object(project,'CACHE',root),patch.object(project,'manifests',return_value=({},models)):
                for command,limit in [(project.server_command(),256),(project.asr_server_command(),64)]:
                    self.assertEqual(command[command.index('--cache-ram')+1],str(limit))
                    self.assertEqual(command[command.index('-c')+1],'2048')
                    self.assertEqual(command[command.index('--host')+1],'127.0.0.1')

    def test_unlimited_or_untyped_budget_rejected_and_zero_disables_cache(self):
        for value in (-1,True,1025,'256',None):
            with self.subTest(value=value),self.assertRaises(ValueError):
                project.prompt_cache_args({'llama':{'prompt_cache_ram_mib':value}},'llama')
        self.assertEqual(project.prompt_cache_args({'llama':{'prompt_cache_ram_mib':0}},'llama'),['--cache-ram','0'])
        self.assertEqual(project.prompt_cache_args({'speech':{}},'asr'),['--cache-ram','64'])
