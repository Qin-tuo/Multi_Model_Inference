"""Verify COCO class mapping, empty-image coverage and missing-image rejection."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from tools.project import sha256
from tools.vision_quality import score


class ScoreWiringTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        annotation=self.root/'gt.json'
        annotation.write_text(json.dumps({'info':{},'images':[{'id':11},{'id':22}],
            'categories':[{'id':1,'name':'person'},{'id':90,'name':'toothbrush'}],
            'annotations':[{'id':1,'image_id':11,'category_id':90,'bbox':[10,20,30,40],
                'area':1200,'iscrowd':0}]}))
        inputs=self.root/'inputs.tsv';inputs.write_text('11\tunused.jpg\n22\tempty.jpg\n')
        dataset=self.root/'dataset.json';dataset.write_text(json.dumps({'ordered_inputs':str(inputs),
            'annotations':{'detection':{'path':str(annotation),'sha256':sha256(annotation)}}}))
        predictions=self.root/'predictions.jsonl'
        predictions.write_text(json.dumps({'image_id':11,'predictions':[{'class_id':1,'score':.9,'bbox':[10,20,30,40]}]})+'\n'+
            json.dumps({'image_id':22,'predictions':[]})+'\n')
        self.args=argparse.Namespace(dataset=dataset,predictions=predictions,task='detection',output=self.root/'metrics.json')

    def test_noncontiguous_coco_category_and_empty_image(self):
        with redirect_stdout(io.StringIO()):self.assertEqual(score(self.args),0)
        result=json.loads(self.args.output.read_text())
        self.assertEqual(result['images'],2)
        self.assertAlmostEqual(result['metrics']['bbox']['AP_percent'],100.,places=8)

    def test_dropping_empty_image_is_rejected(self):
        text=self.args.predictions.read_text().splitlines()[0]
        self.args.predictions.write_text(text+'\n')
        with self.assertRaisesRegex(RuntimeError,'Missing, duplicated or reordered'):
            score(self.args)


if __name__=='__main__':unittest.main(verbosity=2)
