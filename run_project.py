import os
import main_pipeline
import add_missing_data
import visualization

def run_all():
    print("=== Automated Traffic Enforcement Pipeline ===")
    
    # Ask the user which video to process
    video_path = input("Enter the name of your video file (e.g., traffic.mp4): ")
    
    # Safety check: make sure the file actually exists
    if not os.path.exists(video_path):
        print(f"Error: Could not find '{video_path}' in the current folder.")
        return

    try:
        print(f"\n[1/3] Running AI Detection & Tracking on {video_path}...")
        main_pipeline.main(video_path)

        print("\n[2/3] Interpolating Data & Neutralizing Flicker...")
        add_missing_data.main()

        print("\n[3/3] Rendering Final Dashboard Video...")
        visualization.main(video_path)

        print("\n✅ PIPELINE COMPLETE! Open 'final_output.mp4' to view the results.")
        
    except Exception as e:
        print(f"\n❌ Pipeline crashed with error: {e}")

if __name__ == '__main__':
    run_all()