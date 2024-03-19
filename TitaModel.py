import os
import sys
import time
import tqdm
import numpy
import soundfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from calflops import calculate_flops
from encoder_decoder import Encoder, Decoder
from loss import AAMsoftmax
from tools import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf


class TitaNet(nn.Module):
    """
    TitaNet is a neural network for extracting speaker representations,
    by leveraging 1D depth-wise separable convolutions with SE layers
    and a channel attention based statistic pooling layer
    "TitaNet: Neural Model for speaker representation with 1D Depth-wise
    separable convolutions and global context", Kologuri et al.,
    https://arxiv.org/abs/2110.04410
    """

    TARGET_PARAMS = {"s": 6.4, "m": 13.4, "l": 25.3}

    def __init__(self, lr, lr_decay, C, n_class, m, s, test_step,
                 n_mega_blocks=4, n_sub_blocks=3, encoder_output_size=3072, embedding_size=192,
                 encoder_hidden_size=256, mega_block_kernel_sizes=None,
                 prolog_kernel_size=3, epilog_kernel_size=1,
                 attention_hidden_size=128, se_reduction=8, pool_mode='attention_stats',
                 dropout=0.5, device="cpu",
                 **kwargs
                 ):

        super(TitaNet, self).__init__()

        # Define encoder and decoder
        if mega_block_kernel_sizes is None:
            mega_block_kernel_sizes = [3, 7, 11, 15]
        self.encoder = Encoder(
            C,
            n_mega_blocks,
            n_sub_blocks,
            encoder_hidden_size,
            encoder_output_size,
            mega_block_kernel_sizes,
            prolog_kernel_size=prolog_kernel_size,
            epilog_kernel_size=epilog_kernel_size,
            se_reduction=se_reduction,
            dropout=dropout,
        )
        self.decoder = Decoder(
            encoder_output_size,
            attention_hidden_size,
            embedding_size,
            pool_mode=pool_mode,
        )
        # Store loss function
        self.speaker_loss = AAMsoftmax(n_class=n_class, m=m, s=s)
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)
        # self.speaker_encoder = nn.Sequential(
        #     encoder,
        #     decoder
        # )
        self.device = device
        # Transfer to device
        self.to(device)
        para1 = sum(param.numel() for param in self.encoder.parameters()) / 1024 / 1024
        para2 = sum(param.numel() for param in self.decoder.parameters()) / 1024 / 1024
        print(" Model para number = %.2f" % (para1 + para2))

    def forward(self, spectrograms, speakers=None):
        """
        Given input spectrograms of shape [B, M, T], TitaNet returns
        utterance-level embeddings of shape [B, E]
        B: batch size
        M: number of mel frequency bands
        T: maximum number of time steps (frames)
        E: embedding size
        """
        encodings = self.encoder(spectrograms)
        embeddings = self.decoder(encodings)

        # Inference mode
        if speakers is None:
            return F.normalize(embeddings, p=2, dim=1)

        # Training mode
        assert (
                self.loss_function is not None
        ), "Loss function should not be None in training mode"
        return self.loss_function(embeddings, speakers)

    def train_network(self, epoch, loader):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        for num, (data, labels) in enumerate(loader, start=1):
            self.zero_grad()
            labels = torch.LongTensor(labels).to(self.device)
            encodings = self.encoder.forward(data.to(self.device), aug=True)
            speaker_embedding = self.decoder.forward(encodings)
            nloss, prec = self.speaker_loss.forward(speaker_embedding, labels)
            nloss.backward()
            self.optim.step()
            index += len(labels)
            top1 += prec
            loss += nloss.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                             " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * (num / loader.__len__())) +
                             " Loss: %.5f, ACC: %2.2f%% \r" % (loss / (num), top1 / index * len(labels)))
            sys.stderr.flush()
        sys.stdout.write("\n")

        # if epoch % 10 == 0 or epoch == 1:
        #     input_shape = (1, 80, 506)
        #     model = nn.Sequential(
        #         self.encoder,
        #         self.decoder
        #     )
        #     # flops, macs, params = calculate_flops(model=model,
        #     #                                       input_shape=input_shape,
        #     #                                       print_detailed=False,
        #     #                                       output_as_string=True,
        #     #                                       output_precision=4)
        #     # print(f"Model FLOPs: {flops}   MACs: {macs}   Params: {params} \n")
        return loss / num, lr, top1 / index * len(labels)

    def eval_network(self, eval_list, eval_path):
        self.eval()
        files = []
        embeddings = {}
        lines = open(eval_list).read().splitlines()
        for line in lines:
            files.append(line.split()[1])
            files.append(line.split()[2])
        setfiles = list(set(files))
        setfiles.sort()

        for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
            audio, _ = soundfile.read(os.path.join(eval_path, file))
            # Full utterance
            data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).to(self.device)

            # Split utterance matrix
            max_audio = 300 * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = numpy.pad(audio, (0, shortage), 'wrap')
            feats = []
            startframe = numpy.linspace(0, audio.shape[0] - max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf) + max_audio])
            feats = numpy.stack(feats, axis=0).astype(float)
            data_2 = torch.FloatTensor(feats).to(self.device)
            # Speaker embeddings
            with torch.no_grad():
                embedding_1 = self.decoder(self.encoder.forward(data_1, aug=False))
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2 = self.decoder(self.encoder.forward(data_2, aug=False))
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            embeddings[file] = [embedding_1, embedding_2]
        scores, labels = [], []

        for line in lines:
            embedding_11, embedding_12 = embeddings[line.split()[1]]
            embedding_21, embedding_22 = embeddings[line.split()[2]]
            # Compute the scores
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))  # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(line.split()[0]))

        # Compute EER and minDCF
        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

        return EER, minDCF

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)
