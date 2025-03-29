from algorithm import Algorithm
import torch
import torch.nn as nn
from my_env import TEST


class ANN(nn.Module):
    def __init__(self, target_size, mode):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_size = target_size
        self.total_size = 4000
        self.region_count = 125
        self.vector_length = int(self.total_size/self.region_count)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.vector_length, nhead=4, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc_out = nn.Sequential(
            nn.LayerNorm(self.total_size),
            nn.Linear(self.total_size, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = x.view(-1, self.region_count, self.vector_length)
        x = self.transformer(x)
        x = x.flatten(start_dim=1)
        soc_hat = self.fc_out(x)
        soc_hat = soc_hat.reshape(-1)
        return soc_hat


class Algorithm_st1(Algorithm):
    def __init__(self, dataset, train_x, train_y, test_x, test_y, target_size, fold, scaler_y, mode, train_size, reporter, verbose):
        super().__init__(dataset, train_x, train_y, test_x, test_y, target_size, fold, scaler_y, mode, train_size, reporter, verbose)

        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True
        self.indices = torch.linspace(0, 4199, 4000).round().to(torch.int64)

        self.train_x = torch.tensor(train_x[:,self.indices], dtype=torch.float32).to(self.device)
        self.train_y = torch.tensor(train_y, dtype=torch.float32).to(self.device)
        self.test_x = torch.tensor(test_x[:,self.indices], dtype=torch.float32).to(self.device)
        self.test_y = torch.tensor(test_y, dtype=torch.float32).to(self.device)

        self.criterion = torch.nn.MSELoss()
        self.class_size = 1
        self.lr = 0.001
        self.total_epoch = 1000

        if TEST:
            self.total_epoch = 1
            print(test_y.shape)

        self.ann = ANN(self.target_size, mode)
        self.ann.to(self.device)
        self.original_feature_size = self.train_x.shape[1]
        self.reporter.create_epoch_report(self.get_name(), self.dataset.name, self.target_size, self.scaler_y, self.mode, self.train_size, self.fold)
        num_params = sum(p.numel() for p in self.ann.parameters() if p.requires_grad)
        print(num_params)

    def _fit(self):
        self.ann.train()
        self.write_columns()
        optimizer = torch.optim.Adam(self.ann.parameters(), lr=self.lr, weight_decay=self.lr/10)
        for epoch in range(self.total_epoch):
            optimizer.zero_grad()
            y_hat = self.predict_train()
            mse_loss = self.criterion(y_hat, self.train_y)
            loss = mse_loss
            loss.backward()
            optimizer.step()
            if self.verbose:
                self.report(epoch)
        print("|".join([f"{round(i.item() * 4000)}" for i in self.get_indices()]))
        return self

    def predict_train(self):
        return self.ann(self.train_x)

    def predict_test(self):
        return self.ann(self.test_x)

    def write_columns(self):
        if not self.verbose:
            return
        columns = ["epoch","r2","rmse","rpd","rpiq","train_r2","train_rmse","train_rpd","train_rpiq"] + [f"band_{index+1}" for index in range(self.target_size)]
        print("".join([str(i).ljust(20) for i in columns]))

    def report(self, epoch):
        if not self.verbose:
            return
        if epoch%1 != 0:
            return

        bands = list(range(4000))

        train_y_hat = self.predict_train()
        test_y_hat = self.predict_test()

        r2, rmse, rpd, rpiq, r2_o, rmse_o, rpd_o, rpiq_o \
            = self.calculate_metrics(self.test_y, test_y_hat)
        train_r2, train_rmse, train_rpd, train_rpiq, train_r2_o, train_rmse_o, train_rpd_o, train_rpiq_o \
            = self.calculate_metrics(self.train_y, train_y_hat)

        self.reporter.report_epoch_bsdr(epoch, r2, rpd, rpiq, rmse, train_r2, train_rmse, train_rpd, train_rpiq, bands)
        cells = [epoch, r2, rmse, rpd, rpiq, train_r2, train_rmse, train_rpd, train_rpiq] + bands
        cells = [round(item, 5) if isinstance(item, float) else item for item in cells]
        print("".join([str(i).ljust(20) for i in cells]))

    def get_indices(self):
        return self.indices.detach().cpu().numpy().tolist()

    def get_num_params(self):
        return sum(p.numel() for p in self.ann.parameters() if p.requires_grad)
