import pandas as pd

def to_csv(input_file, output_file):
    try:
        # Define column names
        column_names = ['user_id', 'movie_id_ml', 'rating', 'rating_timestamp']
        
        # Read the .dat file with the specified delimiter and column names
        df = pd.read_csv(input_file, delimiter='::', names=column_names, engine='python')
        
        # Save the DataFrame to a .csv file
        df.to_csv(output_file, index=False)
        print(f"File successfully converted to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # Use a raw string or double backslashes for Windows file paths
    to_csv(r'Data\movielens1M.dat', r'Data\movielens1M.csv')
