import pstats
import os

def analyze_profile_data(profile_file_path="./prof.out"):
    """
    Reads a cProfile output file, sorts the statistics by cumulative time,
    and prints the top 20 entries.
    """
    if not os.path.exists(profile_file_path):
        print(f"Error: Profile file not found at {profile_file_path}")
        return

    stats = pstats.Stats(profile_file_path)
    print( stats.sort_stats("cumtime").print_stats(20) )  # pyright: ignore[reportUnusedCallResult]

if __name__ == "__main__":
    analyze_profile_data()
