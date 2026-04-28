import numpy as np

def load_and_dequantize_tessera_representation(representation_file_path, scales_file_path):
    """
    Load and dequantize int8 representations back to float32.

    Args:
        representation_file_path: Path to the int8 representation file (H,W,C)
        scales_file_path: Path to the float32 scales file (H,W)

    Returns:
        representation_f32: float32 ndarray of shape (H,W,C)
    """
    # Load the files
    representation_int8 = np.load(representation_file_path)  # (H, W, C), dtype=int8
    scales = np.load(scales_file_path)  # (H, W), dtype=float32

    # Convert int8 to float32 for computation
    representation_f32 = representation_int8.astype(np.float32)

    # Expand scales to match representation shape
    # scales shape: (H, W) -> (H, W, 1)
    scales_expanded = scales[..., np.newaxis]

    # Dequantize by multiplying with scales
    representation_f32 = representation_f32 * scales_expanded

    return representation_f32

def dequantize_alphaearth_embeddings(values):
    return ((values / 127.5) ** 2) * np.sign(values)
