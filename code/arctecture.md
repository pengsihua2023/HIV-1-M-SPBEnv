## The Arctecture
```
DeepModel(
  (autoencoder): Autoencoder(
    (encoder): Sequential(
      (0): Conv1d(1, 32, kernel_size=(3,), stride=(2,), padding=(1,))
      (1): ReLU()
      (2): ResidualBlock(
        (conv1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        (relu1): ReLU()
        (conv2): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        (relu2): ReLU()
      )
      (3): Conv1d(32, 64, kernel_size=(3,), stride=(2,), padding=(1,))
      (4): ReLU()
      (5): ResidualBlock(
        (conv1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        (relu1): ReLU()
        (conv2): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        (relu2): ReLU()
      )
      (6): Conv1d(64, 128, kernel_size=(3,), stride=(2,), padding=(1,))
      (7): ReLU()
    )
    (decoder): Sequential(
      (0): ConvTranspose1d(128, 64, kernel_size=(3,), stride=(2,), padding=(1,), output_padding=(1,))
      (1): ReLU()
      (2): TransposeResidualBlock(
        (conv1): ConvTranspose1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        (relu1): ReLU()
        (conv2): ConvTranspose1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        (relu2): ReLU()
      (3): ConvTranspose1d(64, 32, kernel_size=(3,), stride=(2,), padding=(1,), output_padding=(1,))
      (4): ReLU()
      (5): TransposeResidualBlock(
        (conv1): ConvTranspose1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        (relu1): ReLU()
        (conv2): ConvTranspose1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        (relu2): ReLU()
      )
      (6): ConvTranspose1d(32, 1, kernel_size=(3,), stride=(2,), padding=(1,), output_padding=(1,))
      (7): ReLU()
    )
  )
  (fc1): Linear(in_features=1048576, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=12, bias=True)
)
```
