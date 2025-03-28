from algorithm import Algorithm
import torch
import torch.nn as nn
from my_env import TEST


class LinearInterpolationModule(nn.Module):
    def __init__(self, y_points, device):
        super(LinearInterpolationModule, self).__init__()
        self.device = device
        self.y_points = y_points.to(device)

    def forward(self, x_new_):
        x_new = x_new_.to(self.device)
        batch_size, num_points = self.y_points.shape
        x_points = torch.linspace(0, 1, num_points).to(self.device).expand(batch_size, -1).contiguous()
        x_new_expanded = x_new.unsqueeze(0).expand(batch_size, -1).contiguous()
        idxs = torch.searchsorted(x_points, x_new_expanded, right=True)
        idxs = idxs - 1
        idxs = idxs.clamp(min=0, max=num_points - 2)
        x1 = torch.gather(x_points, 1, idxs)
        x2 = torch.gather(x_points, 1, idxs + 1)
        y1 = torch.gather(self.y_points, 1, idxs)
        y2 = torch.gather(self.y_points, 1, idxs + 1)
        weights = (x_new_expanded - x1) / (x2 - x1)
        y_interpolated = y1 + weights * (y2 - y1)
        return y_interpolated


class ANN(nn.Module):
    def __init__(self, target_size, mode):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_size = target_size
        self.mode = mode
        init_vals = torch.linspace(0.001, 0.99, self.target_size + 2)
        self.indices = nn.Parameter(torch.tensor([init_vals[i + 1] for i in range(self.target_size)], requires_grad=True).to(self.device))
        if self.mode in ["static", "semi"]:
            self.indices.requires_grad = False
        d_model = 32
        self.embedding = nn.Sequential(nn.Linear(1, d_model), nn.GELU())
        self.register_buffer("pos_encoding", self.get_sinusoidal_encoding(target_size, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc_out = nn.Sequential(
            nn.LayerNorm(target_size * d_model),
            nn.Linear(target_size * d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, linterp):
        outputs = linterp(self.get_indices())
        outputs = outputs.reshape(outputs.shape[0], outputs.shape[1], 1)
        x = self.embedding(outputs)
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = x.flatten(start_dim=1)
        soc_hat = self.fc_out(x)
        soc_hat = soc_hat.reshape(-1)
        return soc_hat

    def get_indices(self):
        return self.indices

    @staticmethod
    def get_sinusoidal_encoding(length, d_model):
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)




class Algorithm_transformer20(Algorithm):
    def __init__(self, dataset, train_x, train_y, test_x, test_y, target_size, fold, scaler_y, mode, train_size, reporter, verbose):
        super().__init__(dataset, train_x, train_y, test_x, test_y, target_size, fold, scaler_y, mode, train_size, reporter, verbose)

        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True

        self.train_x = torch.tensor(train_x, dtype=torch.float32).to(self.device)
        self.train_y = torch.tensor(train_y, dtype=torch.float32).to(self.device)
        self.test_x = torch.tensor(test_x, dtype=torch.float32).to(self.device)
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

        self.linterp_train = LinearInterpolationModule(self.train_x, self.device)
        self.linterp_test = LinearInterpolationModule(self.test_x, self.device)

        self.reporter.create_epoch_report(self.get_name(), self.dataset.name, self.target_size, self.scaler_y, self.mode, self.train_size, self.fold)

    def _fit(self):
        self.ann.train()
        self.write_columns()
        optimizer = torch.optim.Adam(self.ann.parameters(), lr=self.lr, weight_decay=self.lr/10)
        for epoch in range(self.total_epoch):
            if epoch > self.total_epoch/2 and self.mode == "semi":
                self.ann.indices.requires_grad = True
            optimizer.zero_grad()
            y_hat = self.predict_train()
            loss = self.criterion(y_hat, self.train_y)
            loss.backward()
            optimizer.step()
            if self.verbose:
                self.report(epoch)
        print("|".join([f"{round(i.item() * 4200)}" for i in self.ann.indices]))
        return self

    def predict_train(self):
        return self.ann(self.linterp_train)

    def predict_test(self):
        return self.ann(self.linterp_test)

    def write_columns(self):
        if not self.verbose:
            return
        columns = ["epoch","r2","rmse","rpd","rpiq","train_r2","train_rmse","train_rpd","train_rpiq"] + [f"band_{index+1}" for index in range(self.target_size)]
        print("".join([str(i).ljust(20) for i in columns]))

    def report(self, epoch):
        if not self.verbose:
            return
        if epoch%10 != 0:
            return

        bands = self.get_indices()

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
        indices = torch.round(self.ann.get_indices() * self.original_feature_size ).to(torch.int64).tolist()
        return list(dict.fromkeys(indices))

    def get_num_params(self):
        return sum(p.numel() for p in self.ann.parameters() if p.requires_grad)
