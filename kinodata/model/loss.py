class RMSDWeightedLoss:
    def __call__(self, input, target, rmsd):
        weights = 1 / (torch.functional.relu(rmsd) + 1)
        return (weights * (input - target) ** 2) / weights.sum()
