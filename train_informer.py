import argparse
import os
import time
import warnings

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from dataloaders.dataloader_human_behavior import HumanBehaviorDatasetInformer
from defined_models.informer_models.model import Informer

from utils.utils_informer.metrics import metric
from utils.utils_informer.tools import adjust_learning_rate, EarlyStopping

warnings.filterwarnings("ignore")


class Exp():
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        model = Informer(
            self.args.enc_in,
            self.args.dec_in,
            self.args.c_out,
            self.args.seq_len,
            self.args.label_len,
            self.args.pred_len,
            self.args.factor,
            self.args.d_model,
            self.args.n_heads,
            self.args.e_layers,
            self.args.d_layers,
            self.args.d_ff,
            self.args.dropout,
            self.args.attn,
            self.args.activation,
            self.args.output_attention,
            self.args.distil,
            self.args.mix,
            self.device,
        ).float()

        return model

    def _get_data(self, flag):
        if flag == "train":
            shuffle_flag = True
        else:
            shuffle_flag = False

        if flag == "test":
            drop_last = True
        else:
            drop_last = False

        data_set = HumanBehaviorDatasetInformer(
            args=self.args,
            flag=flag,
            )
        print(flag, len(data_set))

        data_loader = DataLoader(
            data_set,
            batch_size=self.args.batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args.num_workers,
            drop_last=drop_last,
        )

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, amsgrad=True)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _process_one_batch(
        self, dataset_object, batch_x, batch_y
    ):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        # decoder input
        if self.args.padding == 0:
            decoder_input = torch.zeros(
                [batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]
            ).float()
        elif self.args.padding == 1:
            decoder_input = torch.ones(
                [batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]
            ).float()
        decoder_input = (
            torch.cat([batch_y[:, : self.args.label_len, :], decoder_input], dim=1)
            .float()
            .to(self.device)
        )

        if self.args.output_attention:
            outputs = self.model(batch_x, decoder_input)[0]
        else:
            outputs = self.model(batch_x, decoder_input)

        true = batch_y[:, -self.args.pred_len:, :].to(self.device)
        return outputs, true

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y
                )

                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 1000 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_post_loss = self.vali(train_data, train_loader, criterion)  # train loss evaluated after training
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_post_loss, vali_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y
            )
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting):
        test_data, test_loader = self._get_data(flag="test")

        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y
            )
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print("test shape:", preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print("test shape:", preds.shape, trues.shape)

        # result save
        folder_path = "./results/informer/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print("mse:{}, mae:{}".format(mse, mae))

        np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + "pred.npy", preds)
        np.save(folder_path + "true.npy", trues)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default="human_behavior", help="data")
    parser.add_argument(
        "--root_path",
        type=str,
        default="/home/zhouyuchen/dumped_human_behavior_data/",
        help="root path of the data file"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="/home/zhouyuchen/feature_encoders/trained_models/informer",
        help="location of model checkpoints",
    )

    parser.add_argument(
        "--seq_len", type=int, default=50, help="input sequence length of Informer encoder"
    )
    parser.add_argument(
        "--label_len", type=int, default=50, help="start token length of Informer decoder"
    )
    parser.add_argument(
        "--pred_len", type=int, default=50, help="prediction sequence length"
    )
    # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

    parser.add_argument("--enc_in", type=int, default=140, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=140, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=140, help="output size")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=4, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=4, help="num of decoder layers")

    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument("--factor", type=int, default=5, help="probsparse attn factor")
    parser.add_argument("--padding", type=int, default=0, help="padding type")
    parser.add_argument(
        "--distil",
        action="store_false",
        help="whether to use distilling in encoder, using this argument means not using distilling",
        default=True,
    )
    parser.add_argument("--dropout", type=float, default=0.05, help="dropout")
    parser.add_argument(
        "--attn",
        type=str,
        default="prob",
        help="attention used in encoder, options:[prob, full]",
    )

    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--output_attention",
        action="store_true",
        help="whether to output attention in encoder",
    )
    parser.add_argument(
        "--mix",
        action="store_false",
        help="use mix attention in generative decoder",
        default=True,
    )

    parser.add_argument(
        "--num_workers", type=int, default=0, help="data loader num workers"
    )
    parser.add_argument("--train_epochs", type=int, default=20, help="train epochs")
    parser.add_argument(
        "--batch_size", type=int, default=256, help="batch size of train input data"
    )
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
    )
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--loss", type=str, default="mse", help="loss function")
    parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate")

    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print("Args in experiment:")
    print(args)

    # setting record of experiments
    setting = "{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_dt{}_mx{}_{}".format(
        args.data,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.attn,
        args.factor,
        args.distil,
        args.mix,
        args.des
    )

    exp = Exp(args)

    print("----------------Start Training: {}----------------".format(setting))
    exp.train(setting)

    print("----------------Testing: {}----------------".format(setting))
    exp.test(setting)


if __name__ == "__main__":
    main()
