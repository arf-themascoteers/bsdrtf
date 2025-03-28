from task_runner import TaskRunner

if __name__ == '__main__':
    train_sizes = [0.75]
    verbose = False
    tag = f"10_best_75"
    tasks = {
        "algorithms": ["bsdrcnn_r"],
        "datasets": ["lucas"],
        "target_sizes": [32],
        "scale_y": ["robust"],
        "mode": ["dyn"],
        "train_sizes": train_sizes
    }
    folds = 1
    ev = TaskRunner(tasks,folds,tag,verbose=verbose)
    ev.evaluate()
