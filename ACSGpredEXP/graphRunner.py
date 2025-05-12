import argparse
import os
from scene_image_analyzer import SceneImageAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Run ACSG scene graph generation on all images in allJPEG directory.")
    parser.add_argument('--openai-key', type=str, help='OpenAI API key (optional)')
    parser.add_argument('--claude-key', type=str, help='Claude API key (optional)')
    args = parser.parse_args()

    # Set up directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    alljpeg_dir = os.path.join(base_dir, 'allJPEG')
    output_dir = os.path.join(base_dir, 'ExpectedVidRun')
    os.makedirs(output_dir, exist_ok=True)

    analyzer = SceneImageAnalyzer(
        api_key=args.openai_key,
        claude_api_key=args.claude_key,
        output_dir=output_dir
    )

    # Loop through all image files in allJPEG
    for fname in os.listdir(alljpeg_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            image_path = os.path.join(alljpeg_dir, fname)
            scene_name = os.path.splitext(fname)[0]
            print(f"\nProcessing {image_path}...")
            analyzer.analyze_image(image_path, scene_name)

if __name__ == "__main__":
    main()
