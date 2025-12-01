from collapse_analysis.base_neural_collapse_analyzer import BaseNeuralCollapseAnalyzer

class LanguageNeuralCollapseAnalyzer(BaseNeuralCollapseAnalyzer):

    def __init__(self, **kwargs):

        super().__init__(kwargs)


    def analyze(
            self,
            model,
            train_loader,
            test_loader,
            ood_loader,
            device
    ):
        """"""
        pass
