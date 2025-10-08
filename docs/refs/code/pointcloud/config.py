class Confing():
    def __init__(self) -> None:
        self.subspace_dim = 3
        # Stride parameters for overlapping patches
        self.stride_x = 1.0  # Default: no overlap
        self.stride_y = 1.0  # Default: no overlap
        #self.interval = 5