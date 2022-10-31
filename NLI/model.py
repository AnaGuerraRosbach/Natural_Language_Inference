import torch
import torch.nn as nn


class SentEncoder(nn.Module):
    def __init__(self, configs, pretrained_emb, token_size, label_size):
        super(SentEncoder, self).__init__()
        """
        Fill this method. You can define and load the word embeddings here.
        You should define the convolution layer here, which use ReLU
        activation. Tips: You can use nn.Sequential to make your code cleaner.
        
        This will be a Siamese network, i.e., the same weights are used to encode
        both the premise and the hypothesis.
        
        """

        self.pretrained_emb = pretrained_emb  # pretrained_emb shape = [16690, 300] (16690 tokens, 300 dimensional embeddings)
        self.conv_dim = configs["conv_dim"]  # conv_dim = [25]
        self.emb_size = pretrained_emb.shape[-1]  # size of each embedding vector = [300]
        self.token_size = token_size  # size of the dictionary of embeddings = [16690]
        self.label_size = label_size  # label_size = [3]

        # EMBEDDING
        self.embed = nn.Embedding.from_pretrained(pretrained_emb, freeze=False)

        #  CONVOLUTIONAL LAYER
        #  The sentence representation will be achieved by using 2 layers of 1D CNN,
        #  to extract representations of the sentences at different level of abstractions.
        #  original implementation: kernel with size of 3, stride of 1, padded with zeros on both sides
        self.conv_layer_1 = nn.Sequential(nn.Conv1d(self.emb_size, self.conv_dim, padding=1, kernel_size=3),
                                          nn.ReLU())

        self.conv_layer_2 = nn.Sequential(nn.Conv1d(self.conv_dim, self.conv_dim, padding=1, kernel_size=3),
                                          nn.ReLU())

    def forward(self, sent):
        """
        Fill this method. It should accept a sentence and return the sentence embeddings
        """
        # shape pretrained emb: [16690, 300] - shape sentence: [16, 50] - batch_size: 16

        emb = self.embed(sent).permute(0, 2, 1)  # from [16, 50, 300] to [16, 300, 50]

        conv_1 = self.conv_layer_1(emb)     # c1: [16,25,50]
        conv_2 = self.conv_layer_2(conv_1)  # c2: [16,25,50]

        # max pooling will reduce the dimensionality (and computational cost) of the convolutional layers
        # at each layer, the representation ui is computed by a max-pooling operation over the feature maps
        u1 = torch.max(conv_1, 2)[0]  # maxpooling first layer  - u1: [16, 50]
        u2 = torch.max(conv_2, 2)[0]  # maxpooling second layer - u2: [16, 50]

        # the final sentence representation u is a concatenation of the representations:
        sent_embs = torch.cat([u1, u2], dim=1)  # sent_embeddings (u1 + u2): [16, 100]

        return sent_embs


class NLINet(nn.Module):
    def __init__(self, configs, pretrained_emb, token_size, label_size):
        super(NLINet, self).__init__()
        """
        Fill this method. You can define the FFNN, dropout and the sentence encoder here.
        sent_encoder: it is a siamese network, i.e., the same weights are used to encode
        both the premise and the hypothesis
        """

        self.hidden = configs["mlp_hidden"]  # hidden_layer = [71]
        self.conv_dim = configs["conv_dim"]  # conv_dim = [25]
        self.encoder = SentEncoder(configs, pretrained_emb, token_size, label_size)  # [16, 50]

        self.layers = nn.Sequential(nn.Linear(8 * self.conv_dim, self.hidden),  # [200, 71]
                                    nn.Dropout(p=0.1),  # Use dropout with p = 0.1 for regularization
                                    nn.Linear(self.hidden, label_size))  # [71, 3]

    def forward(self, premise, hypothesis):
        """
        Fill this method. It should accept a pair of sentence (premise &
        hypothesis) and return the logits.
        ps.: The premise and hypothesis tensors are padded, as we only load
        examples with maximum length of 50 and 1D CNN does not accept inputs
        with varying length.
        """

        u = self.encoder(premise)       # u: [16, 50]
        v = self.encoder(hypothesis)    # v: [16, 50]

        diff = torch.abs(u - v)     # absolute element-wise difference: [16, 50]
        prod = u * v                # element-wise product u ∗ v: [16, 50]

        # concatenation of: concat(u, v) [16, 100], diff(u-v)[16, 50], prod(u*v)[16, 50] = [16, 200]
        x = torch.cat((u, v, diff, prod), dim=1)

        out = self.layers(x)  # ps.: -> no need to apply softmax, since CrossEntropyLoss applies softmax implicitly

        return out
