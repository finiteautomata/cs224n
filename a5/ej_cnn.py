import torch
from torch import nn

if __name__ == "__main__":
    "Ejemplo de CNN"
    batch_len = 100
    channels_in = 20
    channels_out = 5
    kernel_size = 7
    seq_len = 10

    # Nuestra entrada será de tamaño (100, 20, 10)
    # Como el kernel size será de tamaño 7, el nuevo largo será 10 - 7 + 1 = 4
    # Entonces, output será de tamaño (100, 5, 4)
    inp = torch.rand(batch_len, channels_in, seq_len)
    conv_layer = nn.Conv1d(channels_in, channels_out, kernel_size)

    out = conv_layer(inp)

    assert(out.shape == (batch_len, channels_out, 4))

    # Testing max pooling
    # Descarto segundo porque son índices
    max_out, _ = torch.max(out, 2)

    assert(max_out.shape == (batch_len, channels_out))

    print("Tests pasados correctamente")
