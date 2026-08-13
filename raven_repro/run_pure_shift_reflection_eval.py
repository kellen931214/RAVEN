import os
import sys
from pathlib import Path

# Add current worktree and RAVEN to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '/workspace/RAVEN/raven_repro')
sys.path.insert(0, '/workspace/RAVEN')

import json
import torch

from raven_repro.eval import evaluate_clip, evaluate_detector

def main():
    print("PyTorch CUDA device count:", torch.cuda.device_count(), flush=True)
    print("Current CUDA device name:", torch.cuda.get_device_name(0), flush=True)
    metadata_path = Path('/workspace/RAVEN/data/tr/diffusiondb/metadata.csv')
    eval_cfg = {'method': 'TR', 'dataset': 'diffusiondb', 'metadata_path': str(metadata_path)}

    for mag in [24, 32]:
        output_dir = Path(f'/workspace/RAVEN/outputs/pure_pixel_shift_reflection_{mag}px')
        records_path = output_dir / 'records.jsonl'
        if not records_path.is_file():
            print(f'Error: {records_path} does not exist!')
            continue
        records = [json.loads(line) for line in open(records_path)]
        print(f'\n====================================', flush=True)
        print(f'=== Starting Evaluation for {mag}px ({len(records)} records) ===', flush=True)
        print(f'====================================', flush=True)
        
        print(f'--- [1/2] Evaluating CLIP Score for {mag}px ---', flush=True)
        clip_res = evaluate_clip(records, output_dir, device='cuda:0', config=eval_cfg)
        print(f'{mag}px CLIP status:', clip_res.get('status'), 'mean_score:', clip_res.get('mean_score'), flush=True)
        if 'error' in clip_res:
            print(f'{mag}px CLIP error:', clip_res.get('error'), flush=True)
        
        print(f'--- [2/2] Evaluating TR Detector for {mag}px ---', flush=True)
        det_res = evaluate_detector(records, output_dir, method='TR', device='cuda:0', config=eval_cfg)
        tpr = None
        if det_res.get('status') == 'completed' and 'detection_summary' in det_res:
            tpr = det_res['detection_summary'].get('attacked_watermarked_tpr_at_original_threshold')
        print(f'{mag}px Detector status:', det_res.get('status'), 'TPR@1%FPR:', tpr, flush=True)
        if 'setup_error' in det_res:
            print(f'{mag}px Detector setup_error:', det_res.get('setup_error'), flush=True)
        
        summary_path = output_dir / 'results_summary.json'
        data = json.load(open(summary_path))
        data['clip'] = clip_res
        data['tr_detector'] = det_res
        summary_path.write_text(json.dumps(data, indent=2))
        print(f'Successfully updated {mag}px results_summary.json!', flush=True)

if __name__ == '__main__':
    main()
