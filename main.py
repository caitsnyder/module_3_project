from modules.managers.data_manager import DataManager

def main():
    dfs = DataManager(sample_size=5000, run_type_dev=True)
    dfs.get_report()

if __name__ == "__main__":
    main()
