import numpy as np
import torch


class OneHotEncoder:
    def __init__(self, labels=None):
        """Initialize the encoder with an optional set of labels."""
        self.class_to_idx = None  # Mapping from class label to index
        self.idx_to_class = None  # Mapping from index to class label
        self.class_tensor = None  # Tensor of unique class labels (for fast indexing)
        self.color_map = None  # Random color mapping for visualization
        
        if labels is not None:
            self.fit(labels)

    def fit(self, labels):
        """Create a consistent mapping from class labels to indices.
        Ensure that label 0 always maps to index 0.
        """
        unique_classes, _ = torch.sort(torch.unique(labels))  # Get sorted unique class labels
        
        # Ensure label 0 is always mapped to index 0
        if 0 in unique_classes:
            idx_0 = (unique_classes == 0).nonzero().item()  # Find index of 0
            unique_classes = torch.cat([unique_classes[idx_0:idx_0+1], unique_classes[:idx_0], unique_classes[idx_0+1:]])

        self.class_tensor = unique_classes  # Save unique class labels in tensor
        self.class_to_idx = {cls.item(): idx for idx, cls in enumerate(unique_classes)}
        self.idx_to_class = {idx: cls.item() for idx, cls in enumerate(unique_classes)}

        # Generate a color map once fit() is called
        self.generate_color_map()

    def generate_color_map(self):
        """Generate a random color map for each class."""
        if self.class_tensor is None:
            raise ValueError("OneHotEncoder has not been fitted with labels.")

        # Generate random colors for each class, excluding class 0 (background)
        num_classes = len(self.class_tensor)
        np.random.seed(0)  # Fix seed for reproducibility
        colors = np.random.randint(0, 256, size=(num_classes, 3))  # RGB values in range [0, 255]
        
        # Ensure that the background (class 0) has a distinct color (e.g., black)
        colors[0] = [0, 0, 0]
        
        self.color_map = torch.tensor(colors, dtype=torch.uint8)  # Store as tensor for easy indexing

    def label_to_index(self, labels):
        """Convert class labels to indices using tensor-based operations.
        If a label is not in the dictionary, default to index 0.
        """
        if self.class_tensor is None:
            raise ValueError("OneHotEncoder has not been fitted with labels.")

        # Use torch.searchsorted() to efficiently map labels to indices
        indices = torch.searchsorted(self.class_tensor, labels)
        
        # Handle out-of-vocabulary (OOV) labels by mapping them to index 0
        mask = torch.isin(labels, self.class_tensor)  # Check if each label exists in class_tensor
        indices[~mask] = 0  # Default to index 0 for unknown labels
        
        return indices

    def index_to_label(self, indices):
        """Convert indices back to original class labels."""
        if self.class_tensor is None:
            raise ValueError("OneHotEncoder has not been fitted with labels.")

        return self.class_tensor[indices]

    def transform(self, labels):
        """Convert class labels to factored one-hot vectors (four 4-dim vectors, total 16-dim).

        Class index k is base-4 encoded as (k//64, k//16%4, k//4%4, k%4),
        supporting up to 256 classes with 4^4 combinations.
        """
        if self.class_tensor is None:
            raise ValueError("OneHotEncoder has not been fitted with labels.")

        k = self.label_to_index(labels)  # [N], values in [0, num_classes)
        d0 = k // 64        # most significant base-4 digit
        d1 = (k // 16) % 4
        d2 = (k //  4) % 4
        d3 = k % 4          # least significant base-4 digit
        F = torch.nn.functional.one_hot
        return torch.cat([
            F(d0, num_classes=4).to(labels.device),
            F(d1, num_classes=4).to(labels.device),
            F(d2, num_classes=4).to(labels.device),
            F(d3, num_classes=4).to(labels.device),
        ], dim=-1)  # [N, 16]

    def _decode_indices(self, one_hot_labels):
        """Decode factored one-hot [*, 16] back to class indices [*]."""
        d0 = torch.argmax(one_hot_labels[...,  0:4], dim=-1)
        d1 = torch.argmax(one_hot_labels[...,  4:8], dim=-1)
        d2 = torch.argmax(one_hot_labels[..., 8:12], dim=-1)
        d3 = torch.argmax(one_hot_labels[..., 12:16], dim=-1)
        indices = d0 * 64 + d1 * 16 + d2 * 4 + d3
        # Out-of-range indices (e.g. background pixels with near-zero semantics)
        # are clamped to background (index 0).
        indices[indices >= len(self.class_tensor)] = 0
        return indices

    def inverse_transform(self, one_hot_labels):
        """Convert factored one-hot vectors back to class labels."""
        indices = self._decode_indices(one_hot_labels)
        return self.index_to_label(indices)

    def fit_transform(self, labels):
        """Fit the encoder and transform labels to one-hot encoding in one step."""
        self.fit(labels)
        return self.transform(labels)

    def visualize(self, one_hot_labels):
        """Visualize a 2D tensor of class IDs as a color image."""
        if self.color_map is None:
            raise ValueError("Color map has not been generated. Please call fit() first.")

        indices = self._decode_indices(one_hot_labels)
        color_img = self.color_map[indices.long()]
        return color_img

