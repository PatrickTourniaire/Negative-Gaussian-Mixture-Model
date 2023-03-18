import wandb

class HookWandB():

    def start_experiment(self, config: dict):
        wandb.init(
            project="NMMMs",
            config={**config}
        )
    
    def finish_experiment(self):
        wandb.finish()