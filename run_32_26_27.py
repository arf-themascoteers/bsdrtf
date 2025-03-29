from task_runner import TaskRunner

if __name__ == '__main__':
    train_sizes = [0.75]
    verbose = False
    tag = f"run_32_26_27"
    tasks = {
        "algorithms": ["transformer26","transformer27"],
        "datasets": ["lucas"],
        "target_sizes": [32],
        "scale_y": ["robust"],
        "mode": ["dyn"],
        "train_sizes": train_sizes
    }
    folds = 5
    ev = TaskRunner(tasks,folds,tag,verbose=verbose)
    ev.evaluate()
