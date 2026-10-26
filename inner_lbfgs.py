"""Alternative inner optimizer; the learned operator and loss stay unchanged."""
import torch

from ml_ideas import LearnedPrecond


class LBFGSPrecond(LearnedPrecond):
    def _fit(self, params, objective):
        opt = torch.optim.LBFGS(
            params, lr=1.0, max_iter=20, max_eval=40, history_size=10,
            tolerance_grad=1e-7, tolerance_change=1e-9, line_search_fn="strong_wolfe")
        evaluations = 0

        def closure():
            nonlocal evaluations
            opt.zero_grad()
            loss = objective()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite inner L-BFGS training loss")
            loss.backward()
            evaluations += 1
            return loss

        opt.step(closure)
        with torch.no_grad():
            final_loss = objective()
        if not torch.isfinite(final_loss):
            raise FloatingPointError("Nonfinite final inner L-BFGS training loss")
        state = opt.state[params[0]]
        return final_loss, dict(
            optimizer="LBFGS", iterations=state["n_iter"],
            gradient_evaluations=evaluations, loss_evaluations=evaluations + 1,
            nominal_max_eval=40)
