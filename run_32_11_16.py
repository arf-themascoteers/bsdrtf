from task_runner import TaskRunner

if __name__ == '__main__':
    train_sizes = [0.75]
    verbose = True
    tag = f"run_32_11_16"
    tasks = {
        "algorithms": ["transformer11","transformer12","transformer13","transformer14","transformer15","transformer16"],
        "datasets": ["min_lucas"],
        "target_sizes": [32],
        "scale_y": ["robust"],
        "mode": ["dyn"],
        "train_sizes": train_sizes
    }
    folds = 1
    ev = TaskRunner(tasks,folds,tag,verbose=verbose)
    ev.evaluate()
