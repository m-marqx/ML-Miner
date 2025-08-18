class MLStacker:
    def __init__(
        self,
        dataset,
        onchain_data,
        test_index,
        train_in_middle,
        max_trades,
        off_days,
        side,
        cutoff_point,
        best_model_dict: dict,
        verbose: bool = False,
    ):
        self.dataset = dataset
        self.onchain_data = onchain_data
        self.test_index = test_index
        self.train_in_middle = train_in_middle
        self.target_series = self.dataset["Target"]
        self.y_true = self.dataset['Target_bin']
        self.max_trades = max_trades
        self.off_days = off_days
        self.side = side
        self.cutoff_point = cutoff_point
        self.best_model_dict = best_model_dict
        self.verbose = verbose
