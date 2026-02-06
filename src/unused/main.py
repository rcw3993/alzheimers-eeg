import yaml
import argparse
from pathlib import Path
from preprocessing.pipeline import PreprocessingPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config YAML')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output dir
    output_dir = Path('results') / f"{config['name']}_{config['timestamp']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    pipeline = PreprocessingPipeline(config)
    results = pipeline.run(config['subject_ids'])
    
    # Save results
    pipeline.save_results(results, output_dir)
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
