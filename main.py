from modules.managers.data_manager import DataManager

def main():
    dfs = DataManager(sample_size=500, run_type_dev=True)
    # TrainManager()
    dfs.get_report()

if __name__ == "__main__":
    main()
