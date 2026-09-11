from run_experiments import run_all_experiments, save_results
from evaluation import evaluation




def main():
    results = run_all_experiments()
    save_results(results)

    evaluation()
    
    


if __name__ == "__main__":
    main()