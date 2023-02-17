import torch
from pytorch_lightning import Trainer as PLTrainer


class Trainer(PLTrainer):
    # overwrite method because the super method cannot handle non-scalars like lists as metric values
    def _run_evaluate(self):
        assert self.evaluating

        # reload dataloaders
        self._evaluation_loop._reload_evaluation_dataloaders()

        # reset trainer on this loop and all child loops in case user connected a custom loop
        self._evaluation_loop.trainer = self

        with self.profiler.profile(f"run_{self.state.stage}_evaluation"), torch.no_grad():
            eval_loop_results = self._evaluation_loop.run()

        # remove the tensors from the eval results
        for result in eval_loop_results:
            if isinstance(result, dict):
                for k, v in result.items():
                    if isinstance(v, torch.Tensor):
                        if len(v.shape) > 0:
                            result[k] = v.cpu().numpy()
                        else:
                            result[k] = v.cpu().item()

        return eval_loop_results
