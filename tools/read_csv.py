import pandas as pd


class ReadCSV:
    """
    This class is meant to read csv files that contain columns of data,
    including column names, dates, times, numeric values, etc.
    """

    def execute(self, file_path: str):
        """
        Executes the inputted expression that LLM deems is appropriate for
        this tool and returns the answer
        """
        try:
            data = pd.read_csv(file_path)
            return data
        except FileNotFoundError:
            return f"Error: file was not found as {file_path}"
        except Exception as e:
            return f"Error reading file: {e}"
