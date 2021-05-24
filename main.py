from modules.organizers.data_keeper import DataKeeper
def main():
    dfs = DataKeeper(run_type_dev=True)
    dfs.get_report()

if __name__ == "__main__":
    main()
