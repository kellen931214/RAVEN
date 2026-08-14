import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import sys
sys.path.insert(0, '/workspace/RAVEN')

import json
from pathlib import Path

from raven_repro.eval import evaluate_clip, evaluate_detector
from raven_repro.rebuild_eval import rebuild_records, load_metadata

def main():
    print("Environment CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    metadata_path = Path('/workspace/RAVEN/data/tr/diffusiondb/metadata.csv')
    rows = load_metadata(metadata_path)
    eval_cfg = {'method': 'TR', 'dataset': 'diffusiondb', 'metadata_path': str(metadata_path)}

    for mag in [24, 32]:
        output_dir = Path(f'/workspace/RAVEN/outputs/pure_pixel_shift_reflection_{mag}px')
        records = rebuild_records(mag, output_dir, rows)

        print(f'=== [1/2] CLIP Score for {mag}px ===', flush=True)
        clip_res = evaluate_clip(records, output_dir, 'cuda:0', eval_cfg)
        print(f'{mag}px CLIP status:', clip_res.get('status'), 'mean_score:', clip_res.get('mean_score'), flush=True)

        print(f'=== [2/2] TR Detector for {mag}px ===', flush=True)
        det_res = evaluate_detector(records, output_dir, 'TR', 'cuda:0', eval_cfg)
        tpr = None
        if det_res.get('status') == 'completed' and 'detection_summary' in det_res:
            tpr = det_res['detection_summary'].get('attacked_watermarked_tpr_at_original_threshold')
        print(f'{mag}px Detector status:', det_res.get('status'), 'TPR@1%FPR:', tpr, flush=True)

        summary_path = output_dir / 'results_summary.json'
        data = json.load(open(summary_path))
        data['clip'] = clip_res
        data['tr_detector'] = det_res
        summary_path.write_text(json.dumps(data, indent=2))
        print(f'Successfully updated {mag}px results_summary.json!', flush=True)

if __name__ == '__main__':
    main()
