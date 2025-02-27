from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent / 'experiments-outputs'
WARMUP_ITERATIONS = 3

class CsvMerger:
    @staticmethod
    def merge(dir=EXPERIMENT_DIR):
        csv_contents = []
        for csv_file in Path(dir).rglob("GoL-*.out"):
            with open(csv_file, 'r') as file:
                csv_contents.append(file.read())

        header = csv_contents[0].split('\n')[0]
        
        merged_csv_lines = [csv_content.split('\n')[1 + WARMUP_ITERATIONS:] for csv_content in csv_contents]
        flattened_merged_csv_lines = [line.strip() for lines in merged_csv_lines for line in lines if line.strip()]

        final_file = '\n'.join([header] + flattened_merged_csv_lines)

        with open(f"report.csv", 'w') as file:
            file.write(final_file)

if __name__ == "__main__":
    CsvMerger.merge()