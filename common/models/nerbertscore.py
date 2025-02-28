from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from ..types import Entity


class NERBERTScore(nn.Module):
    def __init__(self, model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        """The BERT model adapted for document ranking.

        Args:
            model_name: The name of the model to use.
        """
        super().__init__()
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def forward(
        self,
        true_ents: List[Entity],
        pred_ents: List[Entity],
    ) -> Tuple[float, float, float]:
        """Calculate the semantic similarity between the true and predicted entities.

        Args:
            true_ents: List of true entities.
            pred_ents: List of predicted entities.

        Returns:
            The precision, recall, and F1 score of the semantic similarity.

        """

        ents_text_sim, ents_label_sim = self.get_similarity_tensors(
            true_ents, pred_ents
        )

        # measure the semantic similarity scores
        ents_sim = ents_text_sim * ents_label_sim
        p = torch.mean(torch.max(ents_sim, dim=0).values).item()
        r = torch.mean(torch.max(ents_sim, dim=1).values).item()
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0

        return p, r, f1

    def get_similarity_tensors(
        self,
        true_ents: List[Entity],
        pred_ents: List[Entity],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # get the tensor with semantic similarity of the entity texts
        true_ents_embeds = self.model.encode(
            [ent["text"] for ent in true_ents], normalize_embeddings=True
        )
        pred_ents_embeds = self.model.encode(
            [ent["text"] for ent in pred_ents], normalize_embeddings=True
        )
        ents_text_sim = self.model.similarity(true_ents_embeds, pred_ents_embeds)

        # get the label matching tensor
        ents_label_sim = torch.zeros(len(true_ents), len(pred_ents))
        for i, true_ent in enumerate(true_ents):
            for j, pred_ent in enumerate(pred_ents):
                ents_label_sim[i, j] = true_ent["label"] == pred_ent["label"]

        return ents_text_sim, ents_label_sim

    def visualize(
        self,
        true_ents: List[Entity],
        pred_ents: List[Entity],
    ) -> None:
        def format_text(text, max_len=20):
            _text = text[:max_len]
            if _text != text:
                _text += "..."
            return _text

        # get the input ids of the system and references
        pred_ent_text = [format_text(ent["text"], 20) for ent in pred_ents]
        true_ent_text = [format_text(ent["text"], 20) for ent in true_ents]

        pred_ent_labels = [ent["label"] for ent in pred_ents]
        true_ent_labels = [ent["label"] for ent in true_ents]

        ents_text_sim, ents_label_sim = self.get_similarity_tensors(
            true_ents, pred_ents
        )
        p, r, f1 = self.forward(true_ents, pred_ents)

        # convert to numpy
        ents_text_sim = ents_text_sim.detach().numpy()
        ents_label_sim = ents_label_sim.detach().numpy()

        # initialize the figure object
        fig, axes = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(8.2, 3.6),
            constrained_layout=True,
            squeeze=False,
        )
        axes = axes.flatten()

        # figure preprocessing
        for ax in axes:
            # Turn off axis lines and ticks of the big subplot
            # obs alpha is 0 in RGBA string!
            ax.tick_params(
                labelcolor=(1.0, 1.0, 1.0, 0.0),
                top=False,
                bottom=False,
                left=False,
                right=False,
            )
            # removes the white frame
            ax._frameon = False

        ax_text = fig.add_subplot(1, 2, 1)
        ax_label = fig.add_subplot(1, 2, 2)

        # the cosine distance matrix
        ax_text.set_title("Text", fontsize="large")
        cmim = ax_text.imshow(ents_text_sim, cmap="Blues", vmin=0, vmax=1)
        cbar = fig.colorbar(cmim, ax=ax_text, shrink=0.9)
        cbar.ax.set_ylabel(
            "semantic similarity", rotation=-90, va="bottom", fontsize=12
        )
        # the transport matrix
        ax_label.set_title("Labels", fontsize="large")
        tmim = ax_label.imshow(ents_label_sim, cmap="Greens", vmin=0, vmax=1)
        cbar = fig.colorbar(tmim, ax=ax_label, shrink=1)
        cbar.ax.set_ylabel("label matching", rotation=-90, va="bottom", fontsize=12)

        plot_labels = [
            {
                "plot": ax_text,
                "xticks": pred_ent_text,
                "yticks": true_ent_text,
            },
            {
                "plot": ax_label,
                "xticks": pred_ent_labels,
                "yticks": true_ent_labels,
            },
        ]

        for pl in plot_labels:
            # set the x and y ticks
            pl["plot"].set_yticks(np.arange(len(pl["yticks"])))
            pl["plot"].set_xticks(np.arange(len(pl["xticks"])))

            # add the x and y labels
            pl["plot"].set_yticklabels(pl["yticks"], fontsize=8)
            pl["plot"].set_xticklabels(pl["xticks"], fontsize=8)

            # rotate the x labels a bit
            plt.setp(
                pl["plot"].get_xticklabels(),
                rotation=45,
                ha="right",
                rotation_mode="anchor",
            )

        d_score = round(f1, 3)
        fig.supxlabel(f"F1 Score: {d_score}", fontsize=8)

    def visualize_matrix(
        self,
        true_ents: List[Entity],
        pred_ents: List[Entity],
        data_type: str = "text",
    ) -> None:
        def format_text(text, max_len=20):
            _text = text[:max_len]
            if _text != text:
                _text += "..."
            return _text

        pred_ent = None
        true_ent = None
        if data_type == "text":
            pred_ent = [format_text(ent["text"], 20) for ent in pred_ents]
            true_ent = [format_text(ent["text"], 20) for ent in true_ents]
        elif data_type == "label":
            pred_ent = [ent["label"] for ent in pred_ents]
            true_ent = [ent["label"] for ent in true_ents]
        else:
            raise ValueError(f"Invalid data type: {data_type}")

        ents_text_sim, ents_label_sim = self.get_similarity_tensors(
            true_ents, pred_ents
        )
        p, r, f1 = self.forward(true_ents, pred_ents)

        # convert to numpy
        ents_text_sim = ents_text_sim.detach().numpy()
        ents_label_sim = ents_label_sim.detach().numpy()

        # initialize the figure object
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(6.4, 6.4),
            constrained_layout=True,
        )

        # the cosine distance matrix
        title = "Text" if data_type == "text" else "Label"
        cmap = "Blues" if data_type == "text" else "Greens"
        data = ents_text_sim if data_type == "text" else ents_label_sim
        ylabel = "semantic similarity" if data_type == "text" else "label matching"
        ax.set_title(title, fontsize="large")
        cmim = ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
        cbar = fig.colorbar(cmim, ax=ax, shrink=0.9)
        cbar.ax.set_ylabel(ylabel, rotation=-90, va="bottom", fontsize=12)

        # set the x and y ticks
        ax.set_yticks(np.arange(len(true_ent)))
        ax.set_xticks(np.arange(len(pred_ent)))

        # add the x and y labels
        ax.set_yticklabels(true_ent, fontsize=8)
        ax.set_xticklabels(pred_ent, fontsize=8)

        # rotate the x labels a bit
        plt.setp(
            ax.get_xticklabels(),
            rotation_mode="anchor",
            rotation=45,
            ha="right",
        )

        d_score = round(f1, 3)
        fig.supxlabel(f"F1 Score: {d_score}", fontsize=8)
