## The Arctecture
```
DeepModel(
  (autoencoder): Autoencoder(
    (encoder): Sequential(
      (0): Conv1d(1, 32, kernel_size=(3,), stride=(2,), padding=(1,))
      (1): ReLU()
      (2): ResidualBlock(
        (conv): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        (relu): ReLU()
      )
      (3): Conv1d(32, 64, kernel_size=(3,), stride=(2,), padding=(1,))
      (4): ReLU()
      (5): ResidualBlock(
        (conv): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        (relu): ReLU()
      )
      (6): Conv1d(64, 128, kernel_size=(3,), stride=(2,), padding=(1,))
      (7): ReLU()
      (8): ResidualBlock(
        (conv): Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        (relu): ReLU()
      )
    )
    (decoder): Sequential(
      (0): ConvTranspose1d(128, 64, kernel_size=(3,), stride=(2,), padding=(1,), output_padding=(1,))
      (1): ReLU()
      (2): ConvTranspose1d(64, 32, kernel_size=(3,), stride=(2,), padding=(1,), output_padding=(1,))
      (3): ReLU()
      (4): ConvTranspose1d(32, 1, kernel_size=(3,), stride=(2,), padding=(1,), output_padding=(1,))
      (5): ReLU()
    )
  )
  (fc1): Linear(in_features=65536, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=12, bias=True)
)
```
