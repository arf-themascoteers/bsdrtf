from task_runner import TaskRunner

if __name__ == '__main__':
    train_sizes = [0.75]
    verbose = True
    tag = f"run_128"
    tasks = {
        "algorithms": ["transformer"],
        "datasets": ["lucas"],
        "target_sizes": [128],
        "scale_y": ["robust"],
        "mode": ["dyn"],
        "train_sizes": train_sizes
    }
    folds = 1
    ev = TaskRunner(tasks,folds,tag,verbose=verbose)
    ev.evaluate()
