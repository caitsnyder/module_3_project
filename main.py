from modules.organizers.data_manager import DataManager
def main():
    dfs = DataManager(run_type_dev=True)
    dfs.get_report()

if __name__ == "__main__":
    main()

# https://medium.com/@erikgreenj/k-neighbors-classifier-with-gridsearchcv-basics-3c445ddeb657
# https://medium.com/vickdata/a-simple-guide-to-scikit-learn-pipelines-4ac0d974bdcf
# https://datascience.stackexchange.com/questions/60862/if-i-have-negative-and-positive-numbers-for-a-feature-should-minmaxscaler-be-1