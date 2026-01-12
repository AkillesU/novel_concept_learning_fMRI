import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import filedialog, Tk
import os

def run_sampling_test():
    # 1. File Selection GUI
    root = Tk()
    root.withdraw()  # Hide the main tkinter window
    root.attributes("-topmost", True)
    
    print("Please select a design CSV file to visualize...")
    file_path = filedialog.askopenfilename(
        title="Select Design CSV",
        filetypes=[("CSV Files", "*.csv")]
    )
    root.destroy()

    if not file_path:
        print("No file selected. Exiting.")
        return

    # 2. Load Data
    try:
        df = pd.read_csv(file_path)
        
        # Check for required columns
        required_cols = ['ObjectSpace', 'sampled_f0', 'sampled_f1']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"Error: The selected file is missing columns: {missing}")
            print("Ensure you are using a design file generated with the latest 'create_runs.py'.")
            return
            
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 3. Plotting
    print(f"Plotting sampling data for: {os.path.basename(file_path)}")
    
    plt.figure(figsize=(10, 8))
    
    # Create the scatter plot
    # Each group is assigned a different color/marker
    sns.scatterplot(
        data=df, 
        x='sampled_f0', 
        y='sampled_f1', 
        hue='ObjectSpace', 
        style='ObjectSpace',
        palette='viridis',
        s=100,
        alpha=0.7
    )

    # 4. Formatting the Stimulus Space
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.title(f"Gaussian Sampling Test\nFile: {os.path.basename(file_path)}", fontsize=14)
    plt.xlabel("Feature 0 (Standardized)", fontsize=12)
    plt.ylabel("Feature 1 (Standardized)", fontsize=12)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(title='Object Space (Group)', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save a preview and show the plot
    output_name = f"sampling_test_{os.path.basename(file_path)}.png"
    plt.savefig(output_name)
    print(f"Plot saved as: {output_name}")
    plt.show()

if __name__ == "__main__":
    run_sampling_test()