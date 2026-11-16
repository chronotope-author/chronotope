"""PaddleOCR-VL v1 result bridge to MobileDreamer's original sketch formatter.

The release reads obsolete result attributes. Current official PaddleOCR returns
parsing_res_list blocks. Only this API boundary is adapted; no task metadata,
target detector, or substitute OCR model is used.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .common import SetupError
from .upstream import load_module


def elements_from_result(result):
    if not isinstance(result, dict) or 'parsing_res_list' not in result:
        raise SetupError('Unexpected PaddleOCR-VL result schema; refusing an empty-sketch fallback')
    elements = []
    for block in result['parsing_res_list']:
        if isinstance(block, dict):
            label, text, bbox = block['block_label'], block['block_content'], block['block_bbox']
        else:
            label, text, bbox = block.label, block.content, block.bbox
        if not isinstance(label, str) or not isinstance(text, str) or len(bbox) != 4:
            raise SetupError('Invalid PaddleOCR-VL block')
        elements.append(dict(label=label, text=text, bbox=[int(value) for value in bbox]))
    return elements


class PaddleOCRBridge:
    _pipelines = {}

    def __init__(self, source_root, config, runtime=None):
        module = load_module(source_root, 'mobiledreamer',
            'rollout_imagination_for_GUI_agent/world_model_guided/ocr_module.py')
        self.formatter = module.OCRModule(**config)
        self.config = dict(config)
        self.runtime = dict(runtime or {'pipeline_version': 'v1', 'device': 'cpu',
                                      'enable_new_ir': False, 'cpu_threads': 8})
        if self.runtime.get('pipeline_version') != 'v1':
            raise SetupError('The released MobileDreamer assets require PaddleOCR-VL v1')
        self.pipeline = None

    def initialize(self):
        """Load assets, without calling predict. Shared stateless OCR across cases."""
        key = json.dumps(dict(config=self.config, runtime=self.runtime), sort_keys=True)
        if key not in self._pipelines:
            # Public PaddleX APIs allow its supported legacy IR path. The stock
            # ARM wheel crashes while loading PIR parameters; the official
            # .pdmodel export has byte-identical .pdiparams weights.
            from paddlex import create_pipeline
            from paddlex.inference import load_pipeline_config, PaddlePredictorOption
            config = load_pipeline_config('PaddleOCR-VL')
            config['SubModules']['LayoutDetection'].update(
                model_name=self.config['layout_detection_model_name'],
                model_dir=self.config['layout_detection_model_dir'])
            config['SubModules']['VLRecognition'].update(
                model_name=self.config['vl_rec_model_name'], model_dir=self.config['vl_rec_model_dir'])
            options = PaddlePredictorOption(run_mode='paddle',
                enable_new_ir=self.runtime.get('enable_new_ir', False),
                cpu_threads=self.runtime.get('cpu_threads', 8))
            self._pipelines[key] = create_pipeline(config=config,
                device=self.runtime.get('device', 'cpu'), pp_option=options)
        self.pipeline = self._pipelines[key]
        return self.pipeline

    def image_to_text_description(self, screenshot):
        from PIL import Image
        self.initialize()
        fd, name = tempfile.mkstemp(suffix='.png', prefix='chronotope_mobile_ocr_')
        os.close(fd)
        try:
            image = screenshot if isinstance(screenshot, Image.Image) else Image.fromarray(screenshot)
            image.save(name)
            elements = []
            for result in self.pipeline.predict(name):
                elements.extend(elements_from_result(result))
            return self.formatter._format_elements(elements)
        finally:
            Path(name).unlink(missing_ok=True)
