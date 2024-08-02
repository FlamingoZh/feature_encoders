import os
import sys
import argparse
import numpy as np
import torch
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from defined_models.informer_models.model import Informer

from scipy.stats import zscore

os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/local/bin/ffmpeg"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--label_len", type=int, default=100)
    parser.add_argument("--pred_len", type=int, default=20)
    parser.add_argument("--e_layers", type=int, default=4)
    parser.add_argument("--d_layers", type=int, default=4)
    parser.add_argument("--des", type=str, default="prediction_100ms")
    parser.add_argument("--device", type=str, default="cuda:4")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--target_length", type=int, default=100)

    args = parser.parse_args()

    features = pickle.load(open(f"/home/zhouyuchen/dumped_MELD/data_all_{args.split}.pkl", "rb"))

    new_features = []
    for row in features:
        row = row.swapaxes(0, 1)
        trim_row = row[:args.target_length]
        pad_len = args.target_length - len(trim_row)
        pad_row = np.pad(trim_row, ((0, pad_len), (0, 0)), "constant", constant_values=0)
        new_features.append(pad_row)

    features = new_features

    features = zscore(features, axis=0)

    model = Informer(
        140,
        140,
        140,
        args.seq_len,
        args.label_len,
        args.pred_len,
        5,
        512,
        8,
        args.e_layers,
        args.d_layers,
        2048,
        0.05,
        "prob",
        "gelu",
        False,
        True,
        True,
        "0,1,2,3",
    ).float()

    model.load_state_dict(
        torch.load(
            f"/home/zhouyuchen/feature_encoders/trained_models/informer/human_behavior_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm512_nh8_el{args.e_layers}_dl{args.d_layers}_df2048_atprob_fc5_dtTrue_mxTrue_{args.des}/checkpoint.pth"
        )
    )
    model.eval()
    model.to(args.device);

    batch_size = 256

    latent_representation = []
    model.eval()
    for batch_idx in range(len(features)//batch_size + 1):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(features))
        batch_x = features[start_idx: end_idx]
        batch_x = torch.tensor(batch_x).float().to(args.device)
        with torch.no_grad():
            encoder_output = model.enc_embedding(batch_x)
            encoder_output, attns = model.encoder(encoder_output, attn_mask=None)
        flatten = torch.flatten(encoder_output, start_dim=1, end_dim=2).detach().cpu().numpy()
        for i in flatten:
            latent_representation.append([i])

    latent_representation = np.array(latent_representation).squeeze()
    print("latent_representation shape: ", latent_representation.shape)

    pickle.dump(
        latent_representation, 
        open(f"/home/zhouyuchen/dumped_MELD/latent_representation_all_{args.split}.pkl", "wb")
    )


if __name__ == "__main__":
    main()
