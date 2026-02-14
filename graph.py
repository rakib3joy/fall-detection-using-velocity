import pandas as pd
import matplotlib.pyplot as plt
import os

# Folder where all your CSVs are
csv_folder = "/Users/admin/Desktop/final_velocity/csv_test"

# Loop through all CSV files in the folder
for filename in os.listdir(csv_folder):
    if filename.endswith(".csv"):
        csv_path = os.path.join(csv_folder, filename)

        # Read the CSV
        df = pd.read_csv(csv_path)

        # Create graph_output folder in the same directory
        output_dir = os.path.join(csv_folder, "graph_output")
        os.makedirs(output_dir, exist_ok=True)

        # Prepare output file path
        base_name = os.path.splitext(filename)[0]
        output_file = os.path.join(output_dir, f"{base_name}_graph.png")

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(df["Speed (m/s)"], label="Hip & Shoulder", linestyle='dashed', marker='o')
        plt.title("Fall Detection: Velocity Comparison")
        plt.xlabel("Frame Index")
        plt.ylabel("Velocity (m/s)")
        plt.legend()
        plt.grid()

        # Save and close the plot
        plt.savefig(output_file)
        plt.close()

        print(f"Saved: {output_file}")
