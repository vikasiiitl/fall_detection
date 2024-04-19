import os
import csv

folder_path = '/Users/ajaymaheshwari/Desktop/DEV/AI_Project/Ai-project-5thsem/CNN/Final_0'  # Replace this with the path to your folder
csv_file = 'labels_0.csv'

def create_label_csv(folder_path, csv_file):
    with open(csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['images', 'label'])  # Writing header row
        for filename in os.listdir(folder_path):
            print([filename, '0'])
            csv_writer.writerow([filename, '0'])

create_label_csv(folder_path, csv_file)
print("Done")